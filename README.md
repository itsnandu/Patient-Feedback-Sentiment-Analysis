# Patient Feedback Sentiment Analysis Pipeline

This project implements a complete end-to-end sentiment analysis pipeline for patient survey feedback. It specifically tackles challenges found in messy healthcare survey data, such as capturing negations (e.g., "not rude"), cleaning vendor formatting (HTML tags, emojis), masking staff names, and routing Spanish responses.

## Project Structure
- `generate_large_excel.py`: Synthesizes a 2,100-row dataset mimicking real hospital feedback.
- `data_cleaning.py`: Performs contraction expansion, markup removal, punctuation normalization, and emoji mapping.
- `data_preprocessing.py`: Detects and separates Spanish language responses, and splits contrastive English clauses (e.g., sentences with "but").
- `model_pipeline.py`: Handles feature extraction (Bag-of-n-grams), model training (Multinomial Naive Bayes), and evaluation.

## Prerequisites

Ensure you have Python 3.x installed. You will need to install the following dependencies to run the project:

```bash
pip install pandas openpyxl emoji langdetect scikit-learn joblib streamlit
```

## Steps to Run the Project Freshly

Follow these steps in order to generate the data, clean it, and train the model from scratch. All commands should be run in the terminal from the root directory of the project.

### Step 1: Generate the Synthetic Dataset
Run the data generation script to create the initial raw dataset of 2,100 patient responses.
```bash
python generate_large_excel.py
```
*Output Generated: `patient_feedback_sentiment_large.xlsx`*

### Step 2: Data Cleaning
Run the cleaning script to expand contractions (e.g., "wasn't" -> "was not"), remove HTML tags, normalize punctuation, mask staff names, and convert emojis to text tokens.
```bash
python data_cleaning.py
```
*Output Generated: `patient_feedback_sentiment_cleaned.xlsx`*

### Step 3: Data Pre-processing
Run the pre-processing script to filter out Spanish responses into a separate review queue and apply clause splitting on English responses containing contrastive conjunctions (e.g., "but").
```bash
python data_preprocessing.py
```
*Outputs Generated:*
- *`patient_feedback_sentiment_spanish.xlsx` (Isolated for human review)*
- *`patient_feedback_sentiment_preprocessed.xlsx` (Ready for modeling)*

### Step 4: Feature Extraction, Modeling, and Evaluation
Run the modeling pipeline to vectorize the text using a Bag-of-n-grams approach (capturing both single words and two-word phrases) and train a Multinomial Naive Bayes classifier. The script will output the accuracy and demonstrate that the model successfully interprets negations like "not rude".
```bash
python model_pipeline.py
```
*Outputs Generated:* 
- *`vectorizer_bigram.pkl` (Saved Scikit-Learn vectorizer)*
- *`sentiment_model_nb.pkl` (Saved trained model)*

### Step 5: Run the Web Interface (Optional)
You can interact with the trained model using the provided Streamlit web frontend.
```bash
python -m streamlit run app.py
```
*This will open a web browser where you can type in custom patient feedback (like "the manager was not rude") and see how the model evaluates it in real-time, including how the text is cleaned before scoring.*
