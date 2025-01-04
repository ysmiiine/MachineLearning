import streamlit as st
import pickle
import warnings




# Cache the model loading to improve performance
@st.cache_resource()
def load_model():
    with open('models/voting_model.pkl', 'rb') as f:
        return pickle.load(f)


# Load the preprocessed vectorizer
@st.cache_resource()
def load_vectorizer():
    with open('data/processed/preprocessed_data.pkl', 'rb') as f:
        _, _, vectorizer = pickle.load(f)
    return vectorizer


# Load model and vectorizer
model = load_model()
vectorizer = load_vectorizer()

# Mapping for predictions
label_mapping = {0.0: "Negative", 1.0: "Positive"}

# Streamlit app title
st.title("Sentiment Analysis GUI")

# Input text
input_text = st.text_input("Enter text to analyze:")

if st.button("Predict"):
    if input_text.strip():  # Ensure input is not empty
        # Transform input text
        input_transformed = vectorizer.transform([input_text])  # Transform to TF-IDF vector
        prediction = model.predict(input_transformed)  # Predict sentiment

        # Map the prediction to the corresponding label
        sentiment = label_mapping.get(prediction[0], "Unknown")
        st.write(f"Prediction: {sentiment}")
    else:
        st.error("Please enter some text to analyze.")

warnings.filterwarnings("ignore")
