import os
import string
import pickle
import numpy as np
import pandas as pd
import nltk
from flask import Flask, render_template, request, jsonify

for dep in ['punkt', 'stopwords', 'wordnet', 'punkt_tab']:
    nltk.download(dep, quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

app = Flask(__name__)

rf = None
tfidf = None
scaler = None
summarizer_model = None
summarizer_tokenizer = None
lemmatizer = WordNetLemmatizer()

def load_models():
    global rf, tfidf, scaler, summarizer_model, summarizer_tokenizer
    try:
        with open("model.pkl", "rb") as f:
            rf = pickle.load(f)
        with open("vectorizer.pkl", "rb") as f:
            tfidf = pickle.load(f)
        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        
        print("Loading facebook/bart-large-cnn... This might take a while.")
        model_name = "facebook/bart-large-cnn"
        summarizer_tokenizer = AutoTokenizer.from_pretrained(model_name)
        summarizer_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        print("All models loaded successfully!")
    except Exception as e:
        print(f"Error loading models: {e}")

def count_punct(text):
    words = text.split()
    if len(words) == 0:
        return 0
    punct_count = sum([1 for char in text if char in string.punctuation])
    return round(punct_count / len(words), 3) * 100

def count_cap_words(text):
    words = text.split()
    if len(words) == 0:
        return 0
    cap_count = sum(1 for char in text if char.isupper())
    return round(cap_count / len(words), 3) * 100

def summarize_long_text(text):
    words = text.split()
    chunks = [" ".join(words[i:i+200]) for i in range(0, len(words), 200)]
    summaries = []
    
    for chunk in chunks:
        inputs = summarizer_tokenizer([chunk], max_length=1024, return_tensors='pt', truncation=True)
        summary_ids = summarizer_model.generate(
            inputs['input_ids'],
            num_beams=4,
            min_length=20,
            max_length=80,
            early_stopping=True,
            do_sample=False
        )
        summary = summarizer_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summaries.append(summary)
        
    return " ".join(summaries)


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        if request.is_json:
            news_text = request.json.get("news_text", "")
        else:
            news_text = request.form.get("news_text", "")
        
        if not news_text.strip():
            return jsonify({"error": "Please enter some text to analyze."}), 400
        
        text_clean = "".join([char for char in news_text if char not in string.punctuation]).lower()
        tokens = text_clean.split()
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        text_final = " ".join(tokens)
        
        text_tfidf = tfidf.transform([text_final]).toarray()
        
        body_len = len(news_text) - news_text.count(' ')
        punct = count_punct(news_text)
        caps = count_cap_words(news_text)
        
        features = np.concatenate([text_tfidf[0], [body_len, punct, caps]])
        features_scaled = scaler.transform([features])
        
        prediction = rf.predict(features_scaled)[0]
        confidence = rf.predict_proba(features_scaled).max()
        
        result = {
            "is_real": prediction == 1,
            "confidence": round(confidence * 100, 2),
            "summary": None
        }
        
        if result["is_real"]:
            result["summary"] = summarize_long_text(news_text)
            
        return jsonify({"result": result, "original_text": news_text})

if __name__ == "__main__":
    load_models()
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
