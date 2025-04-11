import joblib
import re

# Optional: Clean function 
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text

# Load model and vectorizer
svm_model = joblib.load('svm_model.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Predict function
def predict_sentence(sentence):
    cleaned = clean_text(sentence)  
    vector = tfidf_vectorizer.transform([cleaned])
    prediction = svm_model.predict(vector)
    return "Positive" if prediction[0] == 1 else "Negative"

# Test
new_sentence = "This movie is amazing! I loved it."
predicted_label = predict_sentence(new_sentence)
print(f"Prediction: {predicted_label}")
