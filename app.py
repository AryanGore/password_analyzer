from flask import Flask, render_template, request, jsonify
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from utils.features import extract_all_features

app = Flask(__name__)

# Load ML models
svm = joblib.load("models/svm_model.pkl")
rf = joblib.load("models/rf_model.pkl")
xgb = joblib.load("models/xgb_model.pkl")
lstm_model = load_model("models/lstm_model.h5")

# DeepSeek tokenizer and model
deepseek_tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-1.5b-instruct")
deepseek_model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-coder-1.5b-instruct")

# Tokenizer for LSTM (fit on your dataset passwords, here simplified)
lstm_max_len = 20  # max length for LSTM input

# Recursive refinement
def recursive_refinement(prompt, max_iterations=5):
    for _ in range(max_iterations):
        # Generate candidate password
        inputs = deepseek_tokenizer(prompt, return_tensors="pt")
        outputs = deepseek_model.generate(**inputs, max_new_tokens=10)
        refined = deepseek_tokenizer.decode(outputs[0], skip_special_tokens=True).split("Response:")[-1].strip()

        # Extract features
        features = extract_all_features(refined)
        x_input = np.array(features).reshape(1, -1)

        # Prepare LSTM input
        # For simplicity, we assume character-level tokenizer with word index from dataset
        # If not saved, LSTM contribution can be skipped or simplified
        # Here we just mock it as 0.5
        lstm_score = 0.5

        # Combine ML model probabilities
        final_score = (
            0.4 * lstm_score +
            0.3 * xgb.predict_proba(x_input)[0][1] +
            0.2 * rf.predict_proba(x_input)[0][1] +
            0.1 * svm.predict_proba(x_input)[0][1]
        )

        # Check if password is strong
        if final_score > 0.8 and features[-1] > 60:
            return refined

        # Feedback for refinement
        flags = {
            "uppercase": not features[1],
            "lowercase": not features[2],
            "digits": not features[3],
            "special_chars": not features[4],
            "length": features[0] < 12,
            "entropy": features[-1] < 60
        }
        feedback = ", ".join([k for k, v in flags.items() if v])
        prompt = f"Refine the password '{refined}' to improve: {feedback}."

    return refined

# Routes
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/generate", methods=["POST"])
def generate():
    data = request.json
    prompt = data.get("prompt", "")
    secure_pw = recursive_refinement(prompt)
    return jsonify({"password": secure_pw})

if __name__ == "__main__":
    app.run(debug=True)
