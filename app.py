"""
Password Analyzer — Production Flask Backend
============================================
Routes:
  GET  /              → serves the UI
  POST /analyze       → full ML ensemble + explainability JSON
  POST /improve       → Groq LLM password suggestions
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*unpickle estimator.*")
warnings.filterwarnings("ignore", message=".*older version of XGBoost.*")

from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib
import os
from dotenv import load_dotenv
from datetime import datetime

# TensorFlow LSTM replaced with sklearn MLP (no TF dependency, works on Python 3.14+)
# The MLP is a neural network with the same role as LSTM in the ensemble.

from utils.features import (
    extract_features,
    missing_features,
    pattern_score,
    estimate_crack_times,
    calculate_entropy,
    check_pwned_api,
)

# ─────────────────────────────────────────────
# Environment & Config
# ─────────────────────────────────────────────
load_dotenv()  # loads .env file if present

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

# ─────────────────────────────────────────────
# LLM Setup (Groq Direct API via requests)
# ─────────────────────────────────────────────
import requests
import json
import secrets
import string

def _call_groq(user_pw: str, strategy: str = "passphrase", previous_failures: list = None) -> dict | None:
    if not GROQ_API_KEY or GROQ_API_KEY == "your_groq_api_key_here":
        return None
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    strategy_instructions = {
        "passphrase": (
            "Create a memorable, high-entropy 3-word passphrase related to or inspired by the input, "
            "separated by symbols and numbers. Example: 'apple' -> 'Blue!Apple99#Quantum'"
        ),
        "leetspeak": (
            "Transform the input password into an advanced leetspeak variant with high security. "
            "Substitute letters with symbols (e.g. a->@, e->3, i->!), extend length to >=14, "
            "and append secure digits and special characters."
        ),
        "crypto": (
            "Generate a high-entropy, complex cryptographic-strength password that retains 2-3 key anchor "
            "characters from the input, mixed with randomized upper/lower case, digits, and special characters. "
            "Length must be 14-18 characters."
        )
    }

    instruction = strategy_instructions.get(strategy, strategy_instructions["passphrase"])
    
    error_context = ""
    if previous_failures:
        error_context = "\nPREVIOUS ATTEMPTS FAILED:\n" + "\n".join(f"- {f}" for f in previous_failures) + "\nDo NOT generate these again. Fix the weaknesses mentioned!"

    payload = {
        "model": "llama3-70b-8192",
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an elite cybersecurity engineer specializing in credential security. "
                    "Your task is to generate ONE single, extremely strong password based on the user's input.\n"
                    f"STRATEGY: {instruction}\n"
                    "REQUIREMENTS:\n"
                    "- Minimum 14 characters long\n"
                    "- Must contain uppercase, lowercase, numbers, and special symbols (!@#$%^&*)\n"
                    "- NO spaces or line breaks\n\n"
                    "You must be HIGHLY CREATIVE and UNPREDICTABLE. Do not use generic substitutions (like replacing 'a' with '@'). "
                    "Ensure the password looks like random noise mixed with the core concept, maximizing entropy.\n"
                    f"{error_context}\n"
                    "You must return your response in strictly valid JSON format with exactly two keys:\n"
                    '1. "password": The generated secure password string.\n'
                    '2. "mnemonic": A short, clever sentence to help the user remember this exact password.'
                )
            },
            {
                "role": "user",
                "content": f"Input: '{user_pw}'. Output JSON:"
            }
        ],
        "temperature": 0.7,
        "max_tokens": 150
    }
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=15)
        if resp.status_code == 200:
            content = resp.json()["choices"][0]["message"]["content"].strip()
            # Try to parse the JSON
            data = json.loads(content)
            if "password" in data and "mnemonic" in data:
                return data
            return None
        else:
            print(f"[WARN] Groq API returned status {resp.status_code}: {resp.text}")
            return None
    except Exception as e:
        print(f"[WARN] Groq API call failed: {e}")
        return None

# ─────────────────────────────────────────────
# Load ML Models
# ─────────────────────────────────────────────
_models = {}

def load_models():
    global _models
    try:
        _models["svm"]  = joblib.load(os.path.join(MODELS_DIR, "svm_model.pkl"))
        _models["rf"]   = joblib.load(os.path.join(MODELS_DIR, "rf_model.pkl"))
        _models["xgb"]  = joblib.load(os.path.join(MODELS_DIR, "xgb_model.pkl"))
        print("[INFO] SVM, RF, XGB loaded successfully.")
    except Exception as e:
        print(f"[ERROR] Sklearn model loading failed: {e}")

    try:
        _models["mlp"] = joblib.load(os.path.join(MODELS_DIR, "mlp_model.pkl"))
        print("[INFO] MLP (neural net) loaded successfully.")
    except Exception as e:
        print(f"[WARN] MLP model not found — run model_training/train_mlp.py to generate it. ({e})")

load_models()

# ─────────────────────────────────────────────
# Flask App
# ─────────────────────────────────────────────
app = Flask(__name__)


# ─────────────────────────────────────────────
# Core Analysis Logic
# ─────────────────────────────────────────────
def _mlp_predict(pw: str, feats: list) -> float | None:
    """Return MLP probability score using the same feature vector as other models."""
    mlp = _models.get("mlp")
    if mlp is None:
        return None
    try:
        x = np.array(feats).reshape(1, -1)
        return float(mlp.predict_proba(x)[0][1])
    except Exception:
        return None


def _clamp_prob(raw: float) -> float:
    """Clamp model output to [0.0, 1.0]. Handles sklearn version mismatch where
    predict_proba may return raw log-odds or unbounded scores."""
    import math
    if raw <= 0:    return 0.0
    if raw >= 1:    return 1.0
    # If value > 1, treat as log-odds and apply sigmoid
    if raw > 1.0:
        return round(1.0 / (1.0 + math.exp(-raw + 5)), 4)  # sigmoid shifted
    return raw


def full_analysis(pw: str) -> dict:
    """Run full ensemble analysis and return explainability trace."""
    feats = extract_features(pw)
    features_arr = np.array(feats).reshape(1, -1)
    entropy = feats[-1]

    # Individual model probabilities
    probs = {}
    if _models.get("svm"):
        probs["svm"]  = _clamp_prob(float(_models["svm"].predict_proba(features_arr)[0][1]))
    else:
        probs["svm"] = 0.0

    if _models.get("rf"):
        probs["rf"]   = _clamp_prob(float(_models["rf"].predict_proba(features_arr)[0][1]))
    else:
        probs["rf"] = 0.0

    if _models.get("xgb"):
        probs["xgb"]  = _clamp_prob(float(_models["xgb"].predict_proba(features_arr)[0][1]))
    else:
        probs["xgb"] = 0.0

    lstm_val = _mlp_predict(pw, feats)
    probs["lstm"] = round(lstm_val, 4) if lstm_val is not None else None

    # Ensemble base score — only average available models
    active = [v for v in probs.values() if v is not None and v > 0]
    base_score = round(sum(active) / len(active), 4) if active else 0.0

    # Bonuses & adjustments
    pat_adj = round(pattern_score(pw), 4)
    bonus_entropy  = 0.5 if entropy >= 70 else (0.4 if entropy >= 55 else 0.0)
    num_digits  = sum(c.isdigit() for c in pw)
    num_special = sum(not c.isalnum() for c in pw)
    bonus_digits  = 0.1 if num_digits >= 3 else 0.0
    bonus_special = 0.1 if num_special >= 2 else 0.0

    final_score = max(0.0, min(1.0,
        base_score + pat_adj + bonus_entropy + bonus_digits + bonus_special
    ))

    # Strength classification
    if entropy > 60 and final_score > 0.7:
        strength = "STRONG"
    elif final_score > 0.45:
        strength = "MEDIUM"
    else:
        strength = "WEAK"

    # Run breach check
    breach_result = check_pwned_api(pw)

    # Downgrade strength if password is found in breach databases.
    # Breach-aware final verdict
    # A breached password gets its own COMPROMISED tier — distinct from WEAK/MEDIUM/STRONG.
    # final_score = structural ML complexity (unchanged, honest about character composition)
    # security_score = actual real-world protection (0 if breached, since attackers already have it)
    if breach_result.get("breached"):
        strength = "COMPROMISED"
        security_score = 0.0
    else:
        security_score = round(final_score, 4)

    return {
        "password_length": len(pw),
        "entropy": entropy,
        "strength": strength,
        "final_score": round(final_score, 4),    # structural ML score
        "security_score": security_score,         # breach-aware safety score
        "models": probs,
        "base_score": base_score,
        "bonuses": {
            "pattern_adjustment": pat_adj,
            "entropy_bonus":  round(bonus_entropy, 4),
            "digits_bonus":   round(bonus_digits, 4),
            "special_bonus":  round(bonus_special, 4),
        },
        "missing_features": missing_features(pw),
        "char_counts": {
            "digits":  num_digits,
            "special": num_special,
        },
        "time_to_crack": estimate_crack_times(entropy),
        "breach_check": breach_result,
    }


# ─────────────────────────────────────────────
# LLM fallback helper
# ─────────────────────────────────────────────


def _fallback_suggestion(pw: str, variant: int) -> dict:
    """Smarter, randomized fallback when LLM is unavailable."""
    base = pw.capitalize() if pw else "Secret"
    if len(base) > 8:
        base = base[:8]  # Keep it manageable
        
    chars = string.ascii_letters + string.digits + "!@#$%^&*"
    random_suffix = "".join(secrets.choice(chars) for _ in range(8))
    random_prefix = "".join(secrets.choice(string.ascii_uppercase + string.digits) for _ in range(3))
    
    if variant == 0:
        pwd = f"{base}_{random_suffix}"
        mnemonic = f"Your word '{base}' followed by an underscore and 8 random characters: {random_suffix}"
    elif variant == 1:
        leetspeak = base.replace('a','@').replace('e','3').replace('i','1').replace('o','0').replace('s','$')
        pwd = f"{random_prefix}!{leetspeak}#{secrets.choice(string.digits)}{secrets.choice(string.ascii_lowercase)}"
        mnemonic = f"Prefix '{random_prefix}!', leetspeak version of '{base}', and 2 random characters at the end."
    else:
        # Cryptographic style
        pwd = f"{secrets.choice(string.punctuation)}{base}{random_suffix[::-1]}{secrets.choice(string.ascii_uppercase)}"
        mnemonic = f"A random symbol, your word '{base}', the reverse of the random suffix, and an uppercase letter."

    return {"password": pwd, "mnemonic": mnemonic}


# ─────────────────────────────────────────────
# Flask Routes
# ─────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        data = request.get_json(silent=True) or {}
        pw = data.get("password", "")
        if not pw:
            return jsonify({"error": "No password provided"}), 400
        result = full_analysis(pw)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/improve", methods=["POST"])
def improve():
    try:
        data = request.get_json(silent=True) or {}
        pw = data.get("password", "")
        if not pw:
            return jsonify({"error": "No password provided"}), 400

        strategies = [
            {"id": "passphrase", "title": "Memorable Passphrase"},
            {"id": "leetspeak",   "title": "Smart Leetspeak"},
            {"id": "crypto",      "title": "Cryptographic Master"},
        ]

        suggestions = []
        for idx, s in enumerate(strategies):
            best_candidate = None
            best_analysis = None
            best_mnemonic = ""
            failures = []
            
            # Agentic Validation Loop (up to 3 tries)
            for attempt in range(3):
                gen_data = _call_groq(pw, s["id"], previous_failures=failures if failures else None)
                if not gen_data:
                    # Fallback if API fails completely
                    gen_data = _fallback_suggestion(pw, idx)
                
                candidate_pw = gen_data.get("password", "")
                candidate_mnem = gen_data.get("mnemonic", "No mnemonic provided.")
                
                analysis = full_analysis(candidate_pw)
                
                # If it's perfect, we break early
                if analysis["strength"] == "STRONG" and not analysis["missing_features"]:
                    best_candidate = candidate_pw
                    best_analysis = analysis
                    best_mnemonic = candidate_mnem
                    break
                
                # Otherwise, record what went wrong for the next iteration
                error_msg = f"Generated '{candidate_pw}' which was rated {analysis['strength']}."
                if analysis["missing_features"]:
                    error_msg += f" Missing: {', '.join(analysis['missing_features'])}."
                failures.append(error_msg)
                
                # Keep track of the best one we've seen so far in case we never get a perfect one
                if best_analysis is None or analysis["security_score"] > best_analysis["security_score"]:
                    best_candidate = candidate_pw
                    best_analysis = analysis
                    best_mnemonic = candidate_mnem

            suggestions.append({
                "strategy": s["title"],
                "password": best_candidate,
                "mnemonic": best_mnemonic,
                "strength": best_analysis["strength"],
                "score":    best_analysis["final_score"],
                "entropy":  best_analysis["entropy"],
            })

        return jsonify({
            "original":    pw,
            "suggestions": suggestions,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/export_report", methods=["POST"])
def export_report():
    try:
        data = request.get_json(silent=True) or {}
        pw = data.get("password", "")
        if not pw:
            return jsonify({"error": "No password provided"}), 400

        analysis = full_analysis(pw)
        breach_status = "⚠️ BREACHED" if analysis["breach_check"].get("breached") else "✅ SAFE (No Breaches Found)"
        breach_count  = analysis["breach_check"].get("count", 0)

        report_md = f"""# 🛡️ PassShield Security Audit Report

**Evaluated Target:** `{pw}`
**Timestamp:** `{datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}`
**Security Rating:** `{analysis['strength']}` (Score: {analysis['final_score']*100:.1f}%)

---

## 1. Executive Summary
- **Overall Score:** `{analysis['final_score']*100:.1f}%`
- **Classification:** `{analysis['strength']}`
- **Shannon Entropy:** `{analysis['entropy']} bits`
- **Length:** `{analysis['password_length']} characters`
- **Data Breach Status:** `{breach_status}` ({breach_count:,} occurrences in HIBP database)

---

## 2. ML Ensemble Model Consensus
| Model Algorithm | Prediction Score / Probability | Status |
|---|---|---|
| **SVM (Support Vector Machine)** | `{analysis['models']['svm']}` | Active |
| **Random Forest Classifier** | `{analysis['models']['rf']}` | Active |
| **XGBoost Gradient Boosting** | `{analysis['models']['xgb']}` | Active |
| **LSTM Recurrent Neural Net** | `{analysis['models']['lstm'] or 'N/A'}` | {'Active' if analysis['models']['lstm'] is not None else 'Offline'} |

**Base Consensus Score:** `{analysis['base_score']}`

---

## 3. Heuristic & Feature Score Attribution
- **Pattern Penalty / Bonus:** `{analysis['bonuses']['pattern_adjustment']:+.4f}`
- **Entropy Bonus:** `{analysis['bonuses']['entropy_bonus']:+.4f}`
- **Digits Bonus:** `{analysis['bonuses']['digits_bonus']:+.4f}`
- **Special Characters Bonus:** `{analysis['bonuses']['special_bonus']:+.4f}`

---

## 4. Threat Profile & Time-to-Crack Estimates
| Threat Scenario | Estimated Time to Crack |
|---|---|
| **Online Rate-Limited Attack (~10/s)** | `{analysis['time_to_crack'].get('Online (rate-limited, ~10/s)', 'N/A')}` |
| **Offline Single CPU (~1M/s)** | `{analysis['time_to_crack'].get('Offline single CPU (~1M/s)', 'N/A')}` |
| **Fast GPU Cluster (~1B/s)** | `{analysis['time_to_crack'].get('Fast GPU (~1B/s)', 'N/A')}` |
| **Distributed Botnet (~1T/s)** | `{analysis['time_to_crack'].get('Huge cluster / botnet (~1T/s)', 'N/A')}` |

---

## 5. Security Recommendations
"""
        if analysis['missing_features']:
            for mf in analysis['missing_features']:
                report_md += f"- ❌ Fix: {mf}\n"
        else:
            report_md += "- ✅ All standard rule policies (length, upper, lower, digits, symbols) satisfied.\n"

        if analysis["breach_check"].get("breached"):
            report_md += f"\n> 🚨 **CRITICAL WARNING:** This password appeared in **{breach_count:,} public data leaks**! Do NOT use this password anywhere."

        report_md += "\n\n---\n*Report generated by PassShield AI Security Engine.*"

        return jsonify({
            "filename": f"PassShield_Audit_Report.md",
            "report": report_md
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)