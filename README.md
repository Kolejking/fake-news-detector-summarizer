# 📰 Fake News Detection & Summarization System

## 🚀 Overview

This project is an end-to-end **NLP-based web application** that detects whether a news article is **Fake or Real** and, if the news is genuine, generates a concise summary using transformer-based models.

It combines **classical machine learning** with **modern deep learning (transformers)** to create a practical, real-world intelligent system.

---

## ✨ Features

* 🔍 **Fake News Detection**

  * Uses TF-IDF vectorization with engineered features
  * Models: Logistic Regression & Random Forest
  * Classifies news as **Fake (0)** or **Real (1)**

* 🧠 **Feature Engineering**

  * Body length of article
  * Punctuation frequency
  * Capital letter frequency

* 🤖 **Abstractive Summarization**

  * Uses BART model via Hugging Face Transformers
  * Generates human-like summaries for real news

* 🌐 **Flask Web Application**

  * User-friendly interface
  * Input news text directly
  * Displays prediction + summary

* 📊 **Model Evaluation**

  * Accuracy, classification report
  * Confusion matrix visualization

---

## 🧠 Tech Stack

* **Programming Language:** Python
* **Machine Learning:** Scikit-learn
* **NLP Techniques:** TF-IDF, Tokenization, Lemmatization
* **Deep Learning:** Hugging Face Transformers (BART)
* **Web Framework:** Flask
* **Libraries:** Pandas, NumPy, NLTK, Matplotlib, Seaborn

---

## ⚙️ How It Works

1. **Input News Article**
2. **Preprocessing**

   * Lowercasing, punctuation removal
   * Stopword removal, lemmatization
3. **Feature Extraction**

   * TF-IDF + engineered features
4. **Classification**

   * Random Forest predicts Fake or Real
5. **Conditional Pipeline**

   * ✅ If REAL → Generate summary using BART
   * ❌ If FAKE → Display warning message

---

## 🧬 Architecture

```
User Input
   ↓
Text Preprocessing
   ↓
TF-IDF + Feature Engineering
   ↓
Random Forest Classifier
   ↓
   ├── Fake → Warning ⚠️
   └── Real → BART Summarizer → Summary 📝
```

---

## ▶️ Run Locally

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/fake-news-detector.git
cd fake-news-detector
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the App

```bash
python main.py
```

### 4️⃣ Open in Browser

```
http://127.0.0.1:5000
```

---

## 📦 Requirements

```
flask
scikit-learn
pandas
numpy
nltk
transformers
torch
matplotlib
seaborn
```

---

## 📊 Dataset

* Fake and real news datasets (CSV format)
* Combined and labeled for supervised learning

---

## 🔥 Future Improvements

* Explainable AI (SHAP) for prediction reasoning
* Source credibility analysis
* Headline vs content consistency check
* Multilingual support
* Deployment (Render / AWS / GCP)

---

## 🏁 Conclusion

This project demonstrates the integration of:

* Traditional ML models for classification
* Transformer-based models for text generation
* Full-stack deployment using Flask

It highlights practical skills in **NLP, machine learning, and real-world system design**.

---

## 👨‍💻 Author

**Subhrajyoti**
Computer Science & Engineering Student

---

## ⭐ If you like this project

Give it a ⭐ on GitHub!
