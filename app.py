import streamlit as st
import joblib
import pandas as pd
import os
from datetime import datetime
from data_cleaning import clean_text

# Load model and vectorizer
@st.cache_resource
def load_models():
    try:
        vectorizer = joblib.load('vectorizer_bigram.pkl')
        model = joblib.load('sentiment_model_nb.pkl')
        return vectorizer, model
    except FileNotFoundError:
        st.error("Model files not found! Please run `python model_pipeline.py` first.")
        return None, None

def main():
    st.set_page_config(
        page_title="Patient Feedback Sentiment", 
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better aesthetics
    st.markdown("""
        <style>
        .stButton>button {
            width: 100%;
            border-radius: 8px;
            font-weight: bold;
            padding: 0.5rem 1rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .result-box-success {
            padding: 20px;
            border-radius: 10px;
            background-color: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        }
        .result-box-error {
            padding: 20px;
            border-radius: 10px;
            background-color: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.title("🏥 About the Model")
        st.info("""
        **Pipeline Overview:**
        - **Vectorization**: Bag-of-n-grams (Unigrams + Bigrams)
        - **Algorithm**: Multinomial Naive Bayes
        - **Goal**: Accurately capture negations in clinical feedback (e.g. *'not rude'*).
        """)
        
        st.markdown("### 💡 Quick Test Examples")
        st.markdown("Click the copy button in the code blocks below to test tricky inputs:")
        st.code("the manager was not rude", language="text")
        st.code("the staff wasn't helpful at all!!! 😠", language="text")
        st.code("Dr. Mike was dismissive and made me feel rushed<br>would not recommend", language="text")
        st.code("the nurses were wonderful, but the wait time was unacceptable", language="text")
    
    # Main Content
    st.title("Patient Feedback Sentiment Analyzer")
    st.markdown("Analyze hospital survey feedback in real-time. Our specialized NLP model automatically handles messy vendor formats, expands contractions, and correctly scores negated phrases.")
    
    vectorizer, model = load_models()
    
    if vectorizer and model:
        st.write("---")
        
        # Input Section
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("📝 Enter Patient Feedback")
            feedback_input = st.text_area("Tell us about your stay:", 
                                          placeholder="e.g., The nurses were wonderful, but the wait time was unacceptable...", 
                                          height=180,
                                          label_visibility="collapsed")
            
            submit_btn = st.button("🔍 Analyze Sentiment", type="primary")
            
        with col2:
            st.subheader("⚙️ Processing Steps")
            st.markdown("""
            When you click analyze, the raw text goes through our automated NLP pipeline:
            1. **Contraction Expansion** 
            2. **Markup Removal**
            3. **Punctuation Normalization**
            4. **Emoji Mapping**
            5. **Staff Name Masking**
            """)
        
        if submit_btn:
            if feedback_input.strip() == "":
                st.warning("⚠️ Please enter some feedback to analyze.")
            else:
                with st.spinner("Analyzing Feedback..."):
                    # 1. Clean the text
                    cleaned_input = clean_text(feedback_input)
                    
                    # 2. Vectorize
                    vectorized_input = vectorizer.transform([cleaned_input])
                    
                    # 3. Predict sentiment
                    prediction = model.predict(vectorized_input)[0]
                    probabilities = model.predict_proba(vectorized_input)[0]
                    
                    # 4. Results Section
                    st.write("---")
                    st.subheader("📊 Analysis Results")
                    
                    res_col1, res_col2 = st.columns([1, 1])
                    
                    with res_col1:
                        if prediction == 'Satisfied':
                            st.markdown(f'<div class="result-box-success"><h1>🟢 Satisfied</h1></div>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<div class="result-box-error"><h1>🔴 Dissatisfied</h1></div>', unsafe_allow_html=True)
                            
                    with res_col2:
                        st.markdown("**Model Confidence Score:**")
                        classes = model.classes_
                        for cls, prob in zip(classes, probabilities):
                            # Use a progress bar for visual impact
                            st.write(f"**{cls}**: {prob:.1%}")
                            st.progress(float(prob))
                            
                    # Show cleaning impact
                    st.write("")
                    with st.expander("🔍 See how the model cleaned your input"):
                        st.markdown("**Original Text:**")
                        st.info(feedback_input)
                        st.markdown("**Cleaned Text (passed to model):**")
                        st.success(cleaned_input)

                    # --- Queue Management ---
                    QUEUE_FILE = "patient_feedback_queue.xlsx"
                    
                    new_entry = pd.DataFrame([{
                        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "Feedback": feedback_input,
                        "Cleaned_Feedback": cleaned_input,
                        "Sentiment": prediction,
                        "Confidence": max(probabilities)
                    }])
                    
                    if os.path.exists(QUEUE_FILE):
                        try:
                            df_queue = pd.read_excel(QUEUE_FILE)
                            df_queue = pd.concat([df_queue, new_entry], ignore_index=True)
                        except Exception as e:
                            # If the file is corrupted or not a valid zip/excel file, start fresh
                            df_queue = new_entry
                    else:
                        df_queue = new_entry
                        
                    # Reorder queue: "Dissatisfied" on top. 
                    # We can use pd.Categorical to ensure 'Dissatisfied' always comes before 'Satisfied'.
                    df_queue['Sentiment'] = pd.Categorical(df_queue['Sentiment'], categories=['Dissatisfied', 'Satisfied'], ordered=True)
                    df_queue = df_queue.sort_values(by=['Sentiment', 'Timestamp'], ascending=[True, False])
                    
                    df_queue.to_excel(QUEUE_FILE, index=False)
                    
                    st.write("---")
                    st.success("✅ Feedback analyzed and added to the queue.")
                    
                    with open(QUEUE_FILE, "rb") as f:
                        st.download_button(
                            label="📥 Download Updated Queue (Excel)",
                            data=f,
                            file_name=QUEUE_FILE,
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )

if __name__ == '__main__':
    main()
