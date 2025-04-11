# NLP-svm-sentiment

# Sentiment Predictor 🧠💬

This project is a simple **Sentiment Analysis** tool built using **Python**, **scikit-learn**, and **NLP techniques**. It takes a sentence as input and predicts whether it's **positive** or **negative**.

---

## 🚀 Features

- Preprocessing of movie review text  
- Text vectorization using **TF-IDF**  
- Sentiment classification using **LinearSVC (SVM)**  
- Easily reusable model for new predictions  
- Model and vectorizer saving with **joblib**

---

## 📁 Project Structure

```
NLP-svm-sentiment/
│
├── svm_model.pkl                # Trained SVM model
├── tfidf_vectorizer.pkl         # Saved TF-IDF vectorizer
├── main.py                      # Script to load model and predict new sentences
├── IMDB Dataset.ipynb             # Notebook to train and save model
├── IMDB Dataset.csv                     # Dataset (e.g., movie reviews)
└── README.md                    # Project documentation
```

---

## 🛠️ How It Works

### 1. Train the Model (save_model.ipynb)

- Load and clean the dataset  
- Vectorize the text using `TfidfVectorizer`  
- Train the model using `LinearSVC`  
- Save the model and vectorizer with `joblib`

### 2. Predict New Sentences (main.py)

- Load the saved model and vectorizer  
- Transform new text with the vectorizer  
- Predict sentiment using the trained SVM model

---


## 🧰 Requirements

- Python 3.7+
- scikit-learn
- pandas
- joblib
- numpy



## 📊 Dataset Used

You can use any binary sentiment dataset. For example:  
[IMDb Movie Reviews (Kaggle)](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews/data)

---

## ✅ Future Improvements

- Add neutral sentiment category  
- Build a web UI with Streamlit or Flask  
- Train deep learning models like LSTM or BERT  
- Add support for more languages  

---

## 🧑‍💻 Author

Made with ❤️ by [yahan madhuhansa]


