import pandas as pd
import re
try:
    from langdetect import detect
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'langdetect'])
    from langdetect import detect

def detect_language_safe(text):
    if not isinstance(text, str) or not text.strip():
        return 'unknown'
    try:
        return detect(text)
    except:
        return 'unknown'

def split_clauses(text):
    if not isinstance(text, str):
        return [text]
    # Split on contrastive conjunctions
    parts = re.split(r'\b(?:but|however|although|though)\b', text, flags=re.IGNORECASE)
    # Clean up whitespace
    return [p.strip() for p in parts if p.strip()]

def main():
    print("Loading cleaned data...")
    df = pd.read_excel('patient_feedback_sentiment_cleaned.xlsx')
    
    print("Detecting language...")
    # Add a column for detected language
    df['detected_lang'] = df['cleaned_feedback'].apply(detect_language_safe)
    
    # Filter out Spanish (or non-English)
    # We will keep 'en' and 'unknown' (short texts might be unknown)
    is_spanish = df['detected_lang'] == 'es'
    es_df = df[is_spanish].copy()
    en_df = df[~is_spanish].copy()
    
    print(f"Found {len(es_df)} Spanish responses. Setting them aside...")
    es_df.to_excel('patient_feedback_sentiment_spanish.xlsx', index=False)
    
    print("Performing clause splitting on English responses...")
    en_df['preprocessed_feedback'] = en_df['cleaned_feedback'].apply(split_clauses)
    
    # Explode the lists into separate rows
    # so "[seg 1]", "[seg 2]" are vectorized independently
    final_df = en_df.explode('preprocessed_feedback').reset_index(drop=True)
    
    print(f"Original English rows: {len(en_df)}")
    print(f"Rows after clause splitting: {len(final_df)}")
    
    final_df.to_excel('patient_feedback_sentiment_preprocessed.xlsx', index=False)
    print("Preprocessed data saved to patient_feedback_sentiment_preprocessed.xlsx")

if __name__ == '__main__':
    main()
