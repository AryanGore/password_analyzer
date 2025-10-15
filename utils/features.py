import re
import math

# Feature extraction functions
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
