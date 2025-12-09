import streamlit as st
import joblib

# Load saved model & vectorizer
@st.cache_resource
def load_model():
    model = joblib.load("my_Fake_news_detection_model.pkl")
    vectorizer = joblib.load("my_vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model()

st.title("Fake News Detector")
st.write("Enter any news article text below to classify it as True or Fake.")

# Text input
text = st.text_area("News Text", height=200)

# Predict button
if st.button("Predict"):
    if text.strip() == "":
        st.warning("Please enter some text.")
    else:
        X = vectorizer.transform([text])
        prediction = model.predict(X)[0]
        proba = model.predict_proba(X)[0]

        # FIXED LOGIC HERE ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
        label = "True News" if prediction == 1 else "Fake News"
        # FIXED ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑

        confidence = round(max(proba) * 100, 2)

        if prediction == 1:
            st.success(f"Prediction: {label} ({confidence}% confidence)")
        else:
            st.error(f"Prediction: {label} ({confidence}% confidence)")
