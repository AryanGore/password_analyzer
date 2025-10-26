# test_models_trace.py
import os
import joblib
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import math, re
from datetime import timedelta

# -------------------------------
# Terminal colors (works on most terminals)
# -------------------------------
class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    CYAN = "\033[96m"
    YELLOW = "\033[93m"
    MAGENTA = "\033[95m"
    RESET = "\033[0m"
    BOLD = "\033[1m"

# -------------------------------
# Paths
# -------------------------------
models_path = "./models"
tokenizer_path = os.path.join(models_path, "lstm_tokenizer.pkl")

# -------------------------------
# Load Models (ensure files exist)
# -------------------------------
svm = joblib.load(os.path.join(models_path, "svm_model.pkl"))
rf = joblib.load(os.path.join(models_path, "rf_model.pkl"))
xgb = joblib.load(os.path.join(models_path, "xgb_model.pkl"))
lstm_model = load_model(os.path.join(models_path, "lstm_model.h5"))

with open(tokenizer_path, "rb") as f:
    tokenizer = joblib.load(f)

# -------------------------------
# Feature / helper functions
# -------------------------------
def password_length(pw): return len(pw)
def has_uppercase(pw): return any(c.isupper() for c in pw)
def has_lowercase(pw): return any(c.islower() for c in pw)
def has_digits(pw): return any(c.isdigit() for c in pw)
def has_special_chars(pw): return any(not c.isalnum() for c in pw)

def calculate_entropy(pw):
    charset = 0
    if re.search(r'[a-z]', pw): charset += 26
    if re.search(r'[A-Z]', pw): charset += 26
    if re.search(r'[0-9]', pw): charset += 10
    if re.search(r'[^a-zA-Z0-9]', pw): charset += 32
    return round(len(pw) * math.log2(charset), 2) if charset > 0 else 0

def extract_features(pw):
    return [
        password_length(pw),
        has_uppercase(pw),
        has_lowercase(pw),
        has_digits(pw),
        has_special_chars(pw),
        calculate_entropy(pw)
    ]

def missing_features(pw):
    missing = []
    if password_length(pw) < 8: missing.append("length < 8")
    if not has_uppercase(pw): missing.append("uppercase missing")
    if not has_lowercase(pw): missing.append("lowercase missing")
    if not has_digits(pw): missing.append("digits missing")
    if not has_special_chars(pw): missing.append("special chars missing")
    return missing

def prepare_lstm_input(password, maxlen=20):
    seq = tokenizer.texts_to_sequences([password])
    return pad_sequences(seq, maxlen=maxlen)

# pattern adjustment (same logic as before)
def pattern_score(password):
    score = 0.0
    common_sequences = ["123", "abc", "password", "qwerty", "admin", "letmein", "welcome"]
    for seq in common_sequences:
        if seq in password.lower():
            score -= 0.3  # strong penalty for very common patterns

    # reward diversity
    if any(c.isupper() for c in password): score += 0.05
    if any(c.islower() for c in password): score += 0.05
    if any(c.isdigit() for c in password): score += 0.05
    if any(not c.isalnum() for c in password): score += 0.05

    # repetition penalty
    if len(password) > 0 and len(set(password)) <= len(password) / 2:
        score -= 0.1

    return max(-0.3, min(0.3, score))

# convert seconds to human readable (years/days/hours/min/sec)
def human_time(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds:.3f} s"
    intervals = [
        ('year', 365*24*3600),
        ('day', 24*3600),
        ('hour', 3600),
        ('minute', 60),
        ('second', 1)
    ]
    parts = []
    remaining = int(round(seconds))
    for name, count in intervals:
        if remaining >= count:
            val = remaining // count
            remaining = remaining % count
            parts.append(f"{val} {name}{'s' if val>1 else ''}")
    return ", ".join(parts) if parts else "0 s"

# time-to-crack estimator using entropy
def estimate_crack_times(entropy_bits: float):
    # Number of possibilities ~= 2^entropy
    possibilities = 2 ** entropy_bits if entropy_bits < 1024 else float('inf')
    # attacker speeds (guesses per second)
    speeds = {
        "Online (rate-limited, ~10/s)": 10,
        "Offline single CPU (~1e6/s)": 1e6,
        "Fast GPU (~1e9/s)": 1e9,
        "Huge cluster / botnet (~1e12/s)": 1e12
    }
    estimates = {}
    for label, gps in speeds.items():
        if possibilities == float('inf'):
            estimates[label] = "infinite"
        else:
            sec = possibilities / gps
            estimates[label] = human_time(sec)
    return estimates

# -------------------------------
# Main ensemble + trace builder
# -------------------------------
def ensemble_predict_with_trace(password):
    feats = extract_features(password)
    features = np.array(feats).reshape(1, -1)
    entropy = feats[-1]
    svm_prob = svm.predict_proba(features)[0][1]
    rf_prob = rf.predict_proba(features)[0][1]
    xgb_prob = xgb.predict_proba(features)[0][1]
    lstm_input = prepare_lstm_input(password)
    lstm_prob = float(lstm_model.predict(lstm_input, verbose=0)[0][0])

    # base ensemble (equal voting or averaged)
    base_score = 0.25 * (svm_prob + rf_prob + xgb_prob + lstm_prob)

    # pattern adjustment
    pat_adj = pattern_score(password)

    # entitlement / bonuses
    bonus_entropy = 0.0
    if entropy >= 70:
        bonus_entropy = 0.5
    elif entropy >= 55:
        bonus_entropy = 0.4

    # digit/special bonuses
    num_digits = sum(c.isdigit() for c in password)
    num_special = sum(not c.isalnum() for c in password)
    bonus_digits = 0.1 if num_digits >= 3 else 0.0
    bonus_special = 0.1 if num_special >= 2 else 0.0

    # sum up
    final_score = base_score + pat_adj + bonus_entropy + bonus_digits + bonus_special
    final_score = max(0.0, min(1.0, final_score))

    # Strength rule: REQUIRE entropy > 60 to consider STRONG.
    # Strength categories: WEAK / MEDIUM / STRONG (but STRONG only if entropy>60 AND final_score>threshold)
    if entropy > 60 and final_score > 0.7:
        strength = "STRONG"
    elif final_score > 0.45:
        strength = "MEDIUM"
    else:
        strength = "WEAK"

    # Build a transparent trace
    trace = {
        "models": {
            "svm_prob": round(float(svm_prob), 4),
            "rf_prob": round(float(rf_prob), 4),
            "xgb_prob": round(float(xgb_prob), 4),
            "lstm_prob": round(float(lstm_prob), 4),
            "base_score": round(base_score, 4)
        },
        "bonuses": {
            "pattern_adj": round(pat_adj, 4),
            "entropy_bonus": round(bonus_entropy, 4),
            "digits_bonus": round(bonus_digits, 4),
            "special_bonus": round(bonus_special, 4)
        },
        "final_score": round(final_score, 4),
        "entropy": entropy,
        "num_digits": num_digits,
        "num_special": num_special,
        "missing_features": missing_features(password),
        "strength": strength
    }

    # time-to-crack table
    trace["time_to_crack"] = estimate_crack_times(entropy)

    return trace

# -------------------------------
# Interactive UI
# -------------------------------
if __name__ == "__main__":
    print(f"{Colors.CYAN}{Colors.BOLD}🔐 Password Analyzer — Ensemble + Trace + Entropy Rule{Colors.RESET}")
    print("Type 'exit' to quit.\n")

    while True:
        pw = input("Enter password to evaluate: ")
        if pw.lower() == "exit":
            break

        trace = ensemble_predict_with_trace(pw)

        # pretty output
        color = Colors.GREEN if trace["strength"] == "STRONG" else (Colors.YELLOW if trace["strength"]=="MEDIUM" else Colors.RED)
        print(f"\n{Colors.MAGENTA}🔎 Evaluation Trace{Colors.RESET}")
        print(f"Password: {Colors.CYAN}{pw}{Colors.RESET}")
        print(f"Entropy : {trace['entropy']} bits")
        print(f"Final Score (0-1): {color}{trace['final_score']}{Colors.RESET}   => {color}{trace['strength']}{Colors.RESET}")
        print("\nModels (probabilities):")
        for k, v in trace["models"].items():
            print(f"  {k:12s}: {v}")
        print("\nBonuses / Adjustments:")
        for k, v in trace["bonuses"].items():
            sign = "+" if v >= 0 else "-"
            print(f"  {k:12s}: {sign}{abs(v)}")
        if trace["missing_features"]:
            print(f"\nMissing / weak rule features: {', '.join(trace['missing_features'])}")
        else:
            print("\nAll basic rule features satisfied ✅")

        print("\nTime-to-crack estimates (approx.):")
        for label, t in trace["time_to_crack"].items():
            print(f"  {label:30s}: {Colors.YELLOW}{t}{Colors.RESET}")

        print("\n" + "-"*70 + "\n")
