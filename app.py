import streamlit as st
import pickle
import re

# Load model & vectorizer
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Function to clean text (make sure this matches training preprocessing)
def clean_text(text):
    text = re.sub(r"http\S+", "", text)  # remove URLs
    text = re.sub(r"[^a-zA-Z]", " ", text)  # remove non-letters
    text = text.lower().strip()
    return text

# Streamlit UI
st.title("📰 AI-Powered Fake News Detector")
st.write("Enter any news text and check if it's *Fake* or *Real*")

# User input
user_input = st.text_area("Paste your news article here:")

if st.button("Check News"):
    if user_input.strip() == "":
        st.warning("⚠ Please enter some text first!")
    else:
        # Preprocess & vectorize
        cleaned = clean_text(user_input)
        vectorized = vectorizer.transform([cleaned])

        # Make prediction
        prediction = model.predict(vectorized)[0]

        # Show result
        if prediction == 0:
            st.error("🚨 This looks like *Fake News*")
        elif prediction == 1:
            st.success("✅ This looks like *Real News*")
        else:
            st.info(f"Model returned: {prediction}")

        # Optional: Show prediction probabilities if supported
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(vectorized)[0]
            st.write(f"🔍 Confidence - Fake: {proba[0]:.2f}, Real: {proba[1]:.2f}")
