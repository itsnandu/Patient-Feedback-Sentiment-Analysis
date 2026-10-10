import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import joblib

def main():
    print("--- Loading Preprocessed Data ---")
    df = pd.read_excel('patient_feedback_sentiment_preprocessed.xlsx')
    
    # We will split data based on the existing dataset_split column
    train_df = df[df['dataset_split'] == 'train'].copy()
    test_df = df[df['dataset_split'] == 'test'].copy()
    
    print(f"Train samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")

    # Clean text columns to make sure no NaNs slip through
    X_train = train_df['preprocessed_feedback'].fillna('')
    y_train = train_df['sentiment_label']
    
    X_test = test_df['preprocessed_feedback'].fillna('')
    y_test = test_df['sentiment_label']

    print("\n--- Step 4: Feature Extraction (Bag-of-n-grams) ---")
    # Initialize unigram+bigram vectorizer (The core requirement for negation)
    vectorizer_bigram = CountVectorizer(ngram_range=(1, 2))
    
    # Initialize unigram-only vectorizer (for comparison in Step 6)
    vectorizer_unigram = CountVectorizer(ngram_range=(1, 1))

    # Fit and transform
    X_train_bigram = vectorizer_bigram.fit_transform(X_train)
    X_test_bigram = vectorizer_bigram.transform(X_test)
    
    X_train_unigram = vectorizer_unigram.fit_transform(X_train)
    X_test_unigram = vectorizer_unigram.transform(X_test)
    
    print(f"Unigram Vocabulary Size: {len(vectorizer_unigram.vocabulary_)}")
    print(f"Bigram (Unigram+Bigram) Vocabulary Size: {len(vectorizer_bigram.vocabulary_)}")
    
    # Save the vectorizer for future use
    joblib.dump(vectorizer_bigram, 'vectorizer_bigram.pkl')
    print("Saved vectorizer to 'vectorizer_bigram.pkl'")

    print("\n--- Step 5: Data Modeling (Multinomial Naive Bayes) ---")
    model_bigram = MultinomialNB()
    model_bigram.fit(X_train_bigram, y_train)
    
    model_unigram = MultinomialNB()
    model_unigram.fit(X_train_unigram, y_train)
    
    # Save the model
    joblib.dump(model_bigram, 'sentiment_model_nb.pkl')
    print("Saved trained model to 'sentiment_model_nb.pkl'")

    print("\n--- Step 6: Data Evaluation ---")
    print("Evaluating Unigram-only Model:")
    y_pred_unigram = model_unigram.predict(X_test_unigram)
    print(f"Accuracy: {accuracy_score(y_test, y_pred_unigram):.4f}")
    
    print("\nEvaluating Unigram+Bigram Model:")
    y_pred_bigram = model_bigram.predict(X_test_bigram)
    print(f"Accuracy: {accuracy_score(y_test, y_pred_bigram):.4f}")
    
    print("\nClassification Report (Unigram+Bigram Model):")
    print(classification_report(y_test, y_pred_bigram))
    
    # Custom Evaluation: Check specific negation-heavy sentences
    print("\n--- Operational Negation Check (Held-out Custom Set) ---")
    from data_cleaning import clean_text
    
    custom_tests = [
        "the manager was not rude",
        "the staff was not helpful at all",
        "Dr. STAFF_NAME was completely unprofessional",
        "the room was not clean"
    ]
    
    cleaned_custom_tests = [clean_text(t) for t in custom_tests]
    
    x_custom_uni = vectorizer_unigram.transform(cleaned_custom_tests)
    x_custom_bi = vectorizer_bigram.transform(cleaned_custom_tests)
    
    preds_uni = model_unigram.predict(x_custom_uni)
    preds_bi = model_bigram.predict(x_custom_bi)
    
    for text, p_uni, p_bi in zip(custom_tests, preds_uni, preds_bi):
        print(f"Text: '{text}'")
        print(f"  -> Unigram Model Predicts : {p_uni}")
        print(f"  -> Bigram Model Predicts  : {p_bi}")
        print()

if __name__ == '__main__':
    main()
