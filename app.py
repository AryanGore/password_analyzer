from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from utils.features import extract_features

# ---------- Flask setup ----------
app = Flask(__name__)

# ---------- Load ML models ----------
svm = pickle.load(open("models/svm_model.pkl", "rb"))
rf = pickle.load(open("models/rf_model.pkl", "rb"))
xgb = pickle.load(open("models/xgb_model.pkl", "rb"))

# Optional small LSTM for sequence scoring
lstm_model = load_model("models/lstm_model.h5")
tokenizer = pickle.load(open("models/tokenizer.pkl", "rb"))

# ---------- DeepSeek API setup ----------
# Example: replace with your actual API endpoint and headers
DEEPSEEK_API_URL = "https://api.deepseek.ai/v1/generate"
DEEPSEEK_API_KEY = "YOUR_API_KEY_HERE"

import requests

def call_deepseek(prompt):
    headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}"}
    payload = {"prompt": prompt, "max_new_tokens": 10}
    response = requests.post(DEEPSEEK_API_URL, json=payload, headers=headers)
    if response.status_code == 200:
        data = response.json()
        # Extract generated text (adjust based on API response structure)
        return data.get("generated_text", "").strip()
    return None

# ---------- Password scoring ----------
def score_password(pw):
    features = np.array(extract_features(pw)).reshape(1, -1)
    x_lstm = pad_sequences(tokenizer.texts_to_sequences([pw]), maxlen=20)
    score = (
        0.4 * lstm_model.predict(x_lstm, verbose=0)[0][0] +
        0.3 * xgb.predict_proba(features)[0][1] +
        0.2 * rf.predict_proba(features)[0][1] +
        0.1 * svm.predict_proba(features)[0][1]
    )
    entropy = extract_features(pw)[-1]
    return score, entropy

# ---------- Mutate/refine password ----------
def refine_password(user_pw, max_iterations=3):
    prompt = f"Refine this password to make it stronger while keeping it similar to the original: {user_pw}"
    refined = user_pw
    for _ in range(max_iterations):
        suggestion = call_deepseek(prompt)
        if not suggestion:
            break
        refined = suggestion
        score, entropy = score_password(refined)
        # Stop if strong enough
        if score > 0.8 and entropy > 60:
            break
        # Update prompt to give feedback
        features = extract_features(refined)
        flags = []
        if not features[1]: flags.append("uppercase")
        if not features[2]: flags.append("lowercase")
        if not features[3]: flags.append("digits")
        if not features[4]: flags.append("special_chars")
        if features[0] < 12: flags.append("length")
        if features[-1] < 60: flags.append("entropy")
        feedback = ", ".join(flags)
        prompt = f"Refine password '{refined}' to improve: {feedback}"
    return refined

# ---------- Flask routes ----------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/improve_password", methods=["POST"])
def improve_password():
    data = request.json
    user_pw = data.get("password", "")
    suggestions = []
    for _ in range(3):  # Generate 3 variants
        refined = refine_password(user_pw)
        score, entropy = score_password(refined)
        suggestions.append({
            "password": refined,
            "strength": "strong" if score > 0.8 else "weak",
            "entropy": entropy
        })
    return jsonify({
        "original": user_pw,
        "suggestions": suggestions
    })

if __name__ == "__main__":
    app.run(debug=True)
