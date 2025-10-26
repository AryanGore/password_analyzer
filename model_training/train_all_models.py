import pandas as pd
import numpy as np
import re
import math
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import joblib
import os
import time
import pickle
import warnings
warnings.filterwarnings("ignore")

# -------------------------------
# Step 1: Paths
# -------------------------------
data_path = "rockyou_100k.csv"  # CSV location
models_path = "../models"        # Where to save models
os.makedirs(models_path, exist_ok=True)

# -------------------------------
# Step 2: Load dataset
# -------------------------------
df = pd.read_csv(data_path)
df.dropna(inplace=True)
df = df[df['password'].apply(lambda x: isinstance(x, str))]

# -------------------------------
# Step 3: Feature extraction functions
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

# Apply feature extraction
df["length"] = df["password"].apply(password_length)
df["uppercase"] = df["password"].apply(has_uppercase)
df["lowercase"] = df["password"].apply(has_lowercase)
df["digits"] = df["password"].apply(has_digits)
df["special_chars"] = df["password"].apply(has_special_chars)
df["entropy"] = df["password"].apply(calculate_entropy)

# -------------------------------
# Step 4: Label creation
# -------------------------------
df["strength"] = df["entropy"].apply(lambda e: "strong" if e > 60 else "weak")

# -------------------------------
# Step 5: Prepare training data
# -------------------------------
X = df[["length", "uppercase", "lowercase", "digits", "special_chars", "entropy"]]
y = df["strength"].map({"weak": 0, "strong": 1})
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------------
# Step 6: Train classical ML models with timing
# -------------------------------
start_time = time.time()
print("Training SVM...")
svm = SVC(kernel='linear', probability=True)
svm.fit(X_train, y_train)
joblib.dump(svm, os.path.join(models_path, "svm_model.pkl"))
print(f"SVM trained in {round(time.time() - start_time, 2)} seconds\n")

start_time = time.time()
print("Training Random Forest...")
rf = RandomForestClassifier(n_jobs=-1)
rf.fit(X_train, y_train)
joblib.dump(rf, os.path.join(models_path, "rf_model.pkl"))
print(f"Random Forest trained in {round(time.time() - start_time, 2)} seconds\n")

start_time = time.time()
print("Training XGBoost...")
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_jobs=-1, verbosity=1)
xgb.fit(X_train, y_train)
joblib.dump(xgb, os.path.join(models_path, "xgb_model.pkl"))
print(f"XGBoost trained in {round(time.time() - start_time, 2)} seconds\n")

# -------------------------------
# Step 7: Train LSTM model with tokenizer saved
# -------------------------------
print("Preparing LSTM data...")
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(df["password"])

# Save the tokenizer for later use
with open(os.path.join(models_path, "lstm_tokenizer.pkl"), "wb") as f:
    pickle.dump(tokenizer, f)

X_lstm = pad_sequences(tokenizer.texts_to_sequences(df["password"]), maxlen=20)
X_lstm_train, X_lstm_test, y_lstm_train, y_lstm_test = train_test_split(X_lstm, y, test_size=0.2, random_state=42)

start_time = time.time()
print("Training LSTM...")
lstm_model = Sequential([
    Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=32, input_length=20),
    LSTM(64),
    Dense(1, activation="sigmoid")
])
lstm_model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
lstm_model.fit(X_lstm_train, y_lstm_train, epochs=3, batch_size=128, verbose=1)
lstm_model.save(os.path.join(models_path, "lstm_model.h5"))
print(f"LSTM trained in {round(time.time() - start_time, 2)} seconds\n")

print("✅ All models and tokenizer trained and saved in the 'models/' folder!")
