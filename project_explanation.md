# Patient Feedback Sentiment Analysis - Complete Project Explanation

This document provides a comprehensive breakdown of the entire Patient Feedback Sentiment Analysis project, including the rationale behind each step and the corresponding Python code.

## 1. Project Overview & Scenario
A hospital network receives daily feedback from discharged patients. The survey contains a 1-5 star rating and an open text box. Historically, staff had to read every review because a patient might leave 4 stars but mention a specific bad interaction in the text (e.g., "room was great, but the nurse was rude").
Due to increasing volume, the goal is to build an NLP model that flags free-text responses as likely `Dissatisfied` so staff can prioritize reading those first. 

**The Core Challenge:** A previous simple model failed because it flagged positive reviews like *"the staff were not rude"* as `Dissatisfied` (it only saw the word "rude"). We must capture **negation** properly.

---

## 2. Data Acquisition (Synthetic Generation)
Since we didn't have the original hospital data, we wrote a script to synthesize 2,100 realistic patient feedback records.

### The Challenge:
We needed the dataset to contain specific edge cases:
- Negations (e.g., "was not") and Contractions (e.g., "wasn't")
- HTML tags from the vendor (`<br>`)
- Repeated punctuation (`!!!`)
- Emojis (`😀`, `😠`)
- Spanish responses
- Contrastive clauses (sentences with "but" or "however")

### The Code (`generate_large_excel.py` snippet):
```python
import pandas as pd
import random

# ... [Setup of templates and vocabularies] ...

def build_dataset(num_samples=2100):
    data = []
    for i in range(num_samples):
        # Assign category (positive, negative, mixed, spanish)
        cat = random.choices(['pos', 'neg', 'mixed', 'spanish'], weights=[0.45, 0.35, 0.15, 0.05])[0]
        
        # Decide if we use contraction or expanded negation
        use_contraction = random.random() < 0.5
        neg_pos = random.choice(["wasn't", "didn't"]) if use_contraction else random.choice(["was not", "did not"])
        
        # ... [Template population logic based on category] ...
        
        data.append({
            'feedback_id': f"PF{i+1:04d}",
            'star_rating': rating,
            'raw_feedback': text,
            'sentiment_label': label,
            'dataset_split': 'train' if random.random() < 0.8 else 'test'
        })
    return pd.DataFrame(data)

df = build_dataset()
df.to_excel('patient_feedback_sentiment_large.xlsx', index=False)
```

---

## 3. Data Cleaning
Raw data is messy. Before we can vectorize the text, we have to clean it.

### The Challenge:
- **Contractions**: "wasn't rude" and "was not rude" mean the same thing, but a machine sees them as different words. We must expand contractions.
- **HTML/Punctuation**: `<br>` tags fuse words together. `!!!` adds meaningless noise.
- **Bias**: If a doctor named "Priya" gets a few bad reviews, the model might learn to associate "Priya" with negative sentiment. We must mask names.

### The Code (`data_cleaning.py`):
```python
import pandas as pd
import re
import emoji

def clean_text(text):
    text = str(text)
    
    # 1. Expand Contractions
    contractions_dict = {"wasn't": "was not", "weren't": "were not", "didn't": "did not"}
    contractions_re = re.compile('(%s)' % '|'.join(contractions_dict.keys()), flags=re.IGNORECASE)
    def replace(match):
        return contractions_dict[match.group(0).lower()]
    text = contractions_re.sub(replace, text)
    
    # 2. Remove Markup
    text = re.sub(r'<br\s*/?>', ' ', text, flags=re.IGNORECASE)
    
    # 3. Normalize Punctuation
    text = re.sub(r'(!)\1+', r'\1', text)
    
    # 4. Mask Entities (Names)
    names_re = re.compile(r'\b(?:Priya|John|Mike|Sarah)\b')
    text = names_re.sub('STAFF_NAME', text)
    
    # 5. Handle Emojis
    text = text.replace('😀', ' EMOJI_POSITIVE ')
    text = text.replace('😠', ' EMOJI_NEGATIVE ')
    text = emoji.replace_emoji(text, replace='')
    
    return text.strip()

# Apply to dataframe
df = pd.read_excel('patient_feedback_sentiment_large.xlsx')
df['cleaned_feedback'] = df['raw_feedback'].apply(clean_text)
```

---

## 4. Data Pre-processing
We need to handle language differences and complex sentence structures.

### The Challenge:
- **Language**: An English model will mis-score Spanish text. We need to isolate Spanish reviews for humans.
- **Contrastive Clauses**: "the nurses were wonderful, but the wait time was unacceptable". If we feed this whole sentence to the model, the positive and negative words cancel each other out. We split on "but".

### The Code (`data_preprocessing.py`):
```python
from langdetect import detect

def split_clauses(text):
    # Split on contrastive conjunctions
    parts = re.split(r'\b(?:but|however|although|though)\b', text, flags=re.IGNORECASE)
    return [p.strip() for p in parts if p.strip()]

df = pd.read_excel('patient_feedback_sentiment_cleaned.xlsx')

# 1. Language Detection
df['lang'] = df['cleaned_feedback'].apply(lambda x: detect(x) if x.strip() else 'unknown')
es_df = df[df['lang'] == 'es'] # Set aside for human review
en_df = df[df['lang'] != 'es']

# 2. Clause Splitting
en_df['preprocessed_feedback'] = en_df['cleaned_feedback'].apply(split_clauses)
# Explode the list so each clause gets its own row for vectorization
final_df = en_df.explode('preprocessed_feedback').reset_index(drop=True)
```

---

## 5. Feature Extraction, Modeling & Evaluation
This is the core Machine Learning step.

### The Challenge:
To solve the "not rude" problem, we use a **Bag-of-n-grams** approach instead of standard Bag-of-Words. By setting `ngram_range=(1, 2)`, the model extracts single words (unigrams: "not", "rude") AND two-word phrases (bigrams: "not rude"). 

### The Code (`model_pipeline.py`):
```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import joblib

# Load data
X_train = train_df['preprocessed_feedback']
y_train = train_df['sentiment_label']
X_test = test_df['preprocessed_feedback']

# Feature Extraction: Unigrams + Bigrams
vectorizer = CountVectorizer(ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Data Modeling: Naive Bayes
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Evaluation
y_pred = model.predict(X_test_vec)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

# Test the core requirement
test_str = "the manager was not rude"
print(model.predict(vectorizer.transform([test_str]))) # Outputs: ['Satisfied']

# Save models for the frontend
joblib.dump(vectorizer, 'vectorizer_bigram.pkl')
joblib.dump(model, 'sentiment_model_nb.pkl')
```

---

## 6. User Interface (Streamlit Frontend)
To make the model usable by hospital staff, we built a web interface.

### The Challenge:
Provide a simple UI where staff can type feedback, see how the text is cleaned under the hood, and view the final Satisfied/Dissatisfied prediction.

### The Code (`app.py` snippet):
```python
import streamlit as st
import joblib
from data_cleaning import clean_text

# Load the saved model and vectorizer
vectorizer = joblib.load('vectorizer_bigram.pkl')
model = joblib.load('sentiment_model_nb.pkl')

st.title("🏥 Patient Feedback Sentiment Analyzer")
feedback_input = st.text_area("Enter Patient Feedback:")

if st.button("Analyze Sentiment"):
    # Clean the input using our existing pipeline
    cleaned_input = clean_text(feedback_input)
    
    # Vectorize and Predict
    vectorized_input = vectorizer.transform([cleaned_input])
    prediction = model.predict(vectorized_input)[0]
    
    if prediction == 'Satisfied':
        st.success("🟢 Satisfied")
    else:
        st.error("🔴 Dissatisfied")
```
