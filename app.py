from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from utils.features import extract_features

import os
import getpass

# ---------- ChatGroq setup ----------
from langchain_groq import ChatGroq

os.environ["GROQ_API_KEY"] = "API_KEY"

llm = ChatGroq(
    model="deepseek-r1-distill-llama-70b",
    temperature=0.2,
    max_tokens=16,
    timeout=10,
    max_retries=2,
)

def call_groq(prompt):
    messages = [
        ("system", (
            "You are a password strength improvement expert. When improving passwords: "
            "1. Keep some similarity to the original password. "
            "2. Ensure minimum length of 12 characters. "
            "3. Include uppercase, lowercase, numbers, and special characters. "
            "4. Maintain readability. "
            "5. Return ONLY the improved password, no explanations. "
            "Example: 'test123' -> 'Test123!@Secure'"
        )),
        ("human", prompt),
    ]
    try:
        ai_msg = llm.invoke(messages)
        print(f"[DEBUG] LLM raw response: {ai_msg}")  # Debug print
        # Try to extract the improved password string
        if hasattr(ai_msg, 'content'):
            if isinstance(ai_msg.content, str):
                result = ai_msg.content.strip()
            elif isinstance(ai_msg.content, list) and len(ai_msg.content) > 0:
                # If it's a list, join or take the first string
                result = str(ai_msg.content[0]).strip()
            else:
                result = str(ai_msg.content).strip()
        else:
            result = str(ai_msg).strip()
        print(f"[DEBUG] Extracted suggestion: {result}")
        return result
    except Exception as e:
        print(f"Error calling Groq API: {e}")
        return None

# ---------- Flask setup ----------
app = Flask(__name__)

# ---------- Load ML models ----------
try:
    # Optional small LSTM for sequence scoring
    lstm_model = load_model("models/lstm_model.h5")
    tokenizer = joblib.load("models/lstm_tokenizer.pkl")  # Updated path
    xgb = joblib.load("models/xgb_model.pkl")
except Exception as e:
    print(f"Error loading models: {e}")
    lstm_model = None
    tokenizer = None
    xgb = None

# ---------- Password scoring ----------
def score_password(pw):
    try:
        features = np.array(extract_features(pw)).reshape(1, -1)
        score = 0.0
        
        if lstm_model is not None and tokenizer is not None:
            x_lstm = pad_sequences(tokenizer.texts_to_sequences([pw]), maxlen=20)
            lstm_score = lstm_model.predict(x_lstm, verbose=0)[0][0]
            score += 0.4 * lstm_score
            
        if xgb is not None:
            xgb_score = xgb.predict_proba(features)[0][1]
            score += 0.6 * xgb_score  # Increased weight since other models might be missing
            
        entropy = extract_features(pw)[-1]
        return score, entropy
    except Exception as e:
        print(f"Error in score_password: {e}")
        return 0.0, 0.0

# ---------- Mutate/refine password ----------
def refine_password(user_pw, max_iterations=3):
    if not user_pw:
        return ""
        
    prompt = (
        f"Make this password stronger while keeping some similarity: {user_pw}\n"
        "Requirements:\n"
        "- Length >= 12 chars\n"
        "- Mix of upper/lowercase\n"
        "- Numbers and special chars\n"
        "- Keep it memorable\n"
        f"- Some similarity to: {user_pw}"
    )
    refined = user_pw
    for _ in range(max_iterations):
        try:
            suggestion = call_groq(prompt)
            print(f"[DEBUG] Suggestion: {suggestion}")  # Debug print
            if suggestion and suggestion != refined:
                refined = suggestion
            else:
                break
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
            if flags:
                feedback = ", ".join(flags)
                prompt = f"Refine password '{refined}' to improve: {feedback}"
            else:
                break
        except Exception as e:
            print(f"Error in refine_password: {e}")
            break
    return refined

# ---------- Flask routes ----------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/improve_password", methods=["POST"])
def improve_password():
    try:
        data = request.json
        user_pw = data.get("password", "")
        if not user_pw:
            return jsonify({"error": "No password provided"}), 400
            
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
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)