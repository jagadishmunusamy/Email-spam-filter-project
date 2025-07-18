import streamlit as st
import pickle

# Load the trained pipeline (vectorizer + model)
with open("spam_classifier_pipeline.pkl", "rb") as f:
    model = pickle.load(f)

# Streamlit App UI
st.title("📩 Email Spam Classifier")
st.write("Enter a message to check whether it's spam or not.")

# Input box
message = st.text_area("Your message:")

# Predict button
if st.button("Predict"):
    if not message.strip():
        st.warning("Please enter a message.")
    else:
        prediction = model.predict([message])[0]
        prob = model.predict_proba([message])[0]

        label = "🚫 Spam" if prediction == 1 else "✅ Not Spam"
        st.success(f"Prediction: {label}")
        st.info(f"Confidence - Spam: {prob[1]*100:.2f}%, Not Spam: {prob[0]*100:.2f}%")