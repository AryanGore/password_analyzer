import re
import math


# ─────────────────────────────────────────────
# Basic feature extractors
# ─────────────────────────────────────────────
def password_length(pw: str) -> int:
    return len(pw)


def has_uppercase(pw: str) -> bool:
    return any(c.isupper() for c in pw)


def has_lowercase(pw: str) -> bool:
    return any(c.islower() for c in pw)


def has_digits(pw: str) -> bool:
    return any(c.isdigit() for c in pw)


def has_special_chars(pw: str) -> bool:
    return any(not c.isalnum() for c in pw)


def calculate_entropy(pw: str) -> float:
    """Shannon-based password entropy in bits."""
    charset = 0
    if re.search(r'[a-z]', pw):        charset += 26
    if re.search(r'[A-Z]', pw):        charset += 26
    if re.search(r'[0-9]', pw):        charset += 10
    if re.search(r'[^a-zA-Z0-9]', pw): charset += 32
    return round(len(pw) * math.log2(charset), 2) if charset > 0 else 0.0


def extract_features(pw: str) -> list:
    """Return feature vector [length, upper, lower, digits, special, entropy]."""
    return [
        password_length(pw),
        int(has_uppercase(pw)),
        int(has_lowercase(pw)),
        int(has_digits(pw)),
        int(has_special_chars(pw)),
        calculate_entropy(pw),
    ]


# ─────────────────────────────────────────────
# Rule-based diagnostics
# ─────────────────────────────────────────────
def missing_features(pw: str) -> list:
    missing = []
    if password_length(pw) < 8:   missing.append("length < 8")
    if not has_uppercase(pw):      missing.append("uppercase missing")
    if not has_lowercase(pw):      missing.append("lowercase missing")
    if not has_digits(pw):         missing.append("digits missing")
    if not has_special_chars(pw):  missing.append("special chars missing")
    return missing


# ─────────────────────────────────────────────
# Pattern-based score adjustment
# ─────────────────────────────────────────────
COMMON_PATTERNS = [
    "123", "1234", "12345", "123456", "654321",
    "abc", "abcd", "qwerty", "password", "pass",
    "admin", "letmein", "welcome", "monkey", "iloveyou",
    "dragon", "master", "sunshine", "shadow", "hello",
]


def pattern_score(pw: str) -> float:
    """Returns a float in [-0.3, +0.2] as an ensemble adjustment."""
    score = 0.0
    lower_pw = pw.lower()
    for seq in COMMON_PATTERNS:
        if seq in lower_pw:
            score -= 0.25

    # Reward character diversity
    if any(c.isupper() for c in pw):     score += 0.05
    if any(c.islower() for c in pw):     score += 0.05
    if any(c.isdigit() for c in pw):     score += 0.05
    if any(not c.isalnum() for c in pw): score += 0.05

    # Repetition penalty
    if len(pw) > 0 and len(set(pw)) <= len(pw) / 2:
        score -= 0.10

    return max(-0.3, min(0.2, score))


# ─────────────────────────────────────────────
# Time-to-crack estimation
# ─────────────────────────────────────────────
ATTACKER_SPEEDS = {
    "Online (rate-limited, ~10/s)":        10,
    "Offline single CPU (~1M/s)":           1_000_000,
    "Fast GPU (~1B/s)":                    1_000_000_000,
    "Huge cluster / botnet (~1T/s)":       1_000_000_000_000,
}


def human_time(seconds: float) -> str:
    if seconds == float('inf'):
        return "∞ (effectively forever)"
    if seconds < 1:
        return f"{seconds:.4f} seconds"
    intervals = [
        ('century',  100 * 365.25 * 24 * 3600),
        ('year',     365.25 * 24 * 3600),
        ('day',      24 * 3600),
        ('hour',     3600),
        ('minute',   60),
        ('second',   1),
    ]
    parts = []
    remaining = float(seconds)
    for name, count in intervals:
        if remaining >= count:
            val = int(remaining // count)
            remaining %= count
            parts.append(f"{val:,} {name}{'s' if val > 1 else ''}")
            if len(parts) >= 2:
                break
    return ", ".join(parts) if parts else "< 1 second"


def estimate_crack_times(entropy_bits: float) -> dict:
    possibilities = 2 ** min(entropy_bits, 1023)
    result = {}
    for label, gps in ATTACKER_SPEEDS.items():
        seconds = possibilities / gps
        result[label] = human_time(seconds)
    return result


import hashlib
import requests

# ─────────────────────────────────────────────
# Zero-Knowledge Breach Check (HaveIBeenPwned k-Anonymity Protocol)
# ─────────────────────────────────────────────
def check_pwned_api(pw: str) -> dict:
    """
    Checks if password exists in known data breaches using k-Anonymity protocol.
    Only the first 5 characters of the SHA-1 hash are sent over the network.
    Zero privacy leak.
    """
    if not pw:
        return {"breached": False, "count": 0}
    try:
        sha1_hash = hashlib.sha1(pw.encode('utf-8')).hexdigest().upper()
        prefix = sha1_hash[:5]
        suffix = sha1_hash[5:]

        url = f"https://api.pwnedpasswords.com/range/{prefix}"
        headers = {"User-Agent": "PassShield-Security-Analyzer"}
        resp = requests.get(url, headers=headers, timeout=5)

        if resp.status_code == 200:
            hashes = (line.split(':') for line in resp.text.splitlines())
            for h, count in hashes:
                if h == suffix:
                    return {"breached": True, "count": int(count)}
            return {"breached": False, "count": 0}
        else:
            return {"breached": False, "count": 0, "error": f"API status {resp.status_code}"}
    except Exception as e:
        return {"breached": False, "count": 0, "error": str(e)}
