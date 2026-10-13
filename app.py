
import streamlit as st
import joblib
from pathlib import Path

st.set_page_config(
    page_title="SMS Spam Detection",
    page_icon="📩",
    layout="centered"
)

@st.cache_resource
def load_model():
    folder = Path(__file__).resolve().parent
    model = joblib.load(folder / "spam_model.pkl")
    vectorizer = joblib.load(folder / "tfidf_vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model()

st.title("📩 SMS Spam Detection System")
st.write("Machine Learning Capstone Project 2")
st.write("This application uses NLP and Logistic Regression to classify SMS messages.")

st.divider()

message = st.text_area(
    "Enter your SMS message:",
    placeholder="Type or paste your message here...",
    height=150
)

if st.button("Predict Message", type="primary"):
    if not message.strip():
        st.warning("Please enter a message first.")
    else:
        features = vectorizer.transform([message])
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0]

        if prediction == 1:
            st.error("🚨 SPAM MESSAGE DETECTED")
        else:
            st.success("✅ HAM - LEGITIMATE MESSAGE")

        st.write(
            f"Spam Probability: {probability[1] * 100:.2f}%"
        )

st.divider()
st.caption("Developed by Himani Verma | Capstone Project 2")
