# ⚙️ System Design & Technical Specifications

## 1. System Specifications & Performance SLA

- **Response Latency (Analysis Endpoint):** $< 80\text{ms}$ (excluding remote network API calls)
- **Response Latency (HIBP Breach Check):** $< 300\text{ms}$ (cached DNS / SSL connection)
- **Response Latency (Groq LLM Generation):** $< 1.2\text{s}$ (DeepSeek R1 via Groq LPUs)
- **UI Animation Performance:** $60\text{ fps}$ (Canvas API GPU accelerated rendering)
- **Memory Footprint:** $\approx 180\text{ MB}$ (Flask runtime + loaded Scikit-Learn/XGBoost `.pkl` models)

---

## 2. API Contract Specifications

### 2.1 `POST /analyze`
- **Request Body:**
```json
{
  "password": "string"
}
```

- **Response Body (`200 OK`):**
```json
{
  "password_length": 16,
  "entropy": 104.87,
  "strength": "STRONG",
  "final_score": 1.0,
  "base_score": 1.0,
  "models": {
    "svm": 1.0,
    "rf": 1.0,
    "xgb": 0.9999,
    "lstm": null
  },
  "bonuses": {
    "pattern_adjustment": 0.2,
    "entropy_bonus": 0.5,
    "digits_bonus": 0.1,
    "special_bonus": 0.1
  },
  "missing_features": [],
  "char_counts": { "digits": 4, "special": 2 },
  "time_to_crack": {
    "Online (rate-limited, ~10/s)": "1,174,658,642,118,018,203,648 centurys, 41 years",
    "Offline single CPU (~1M/s)": "11,746,586,421,180,180 centurys, 70 years",
    "Fast GPU (~1B/s)": "11,746,586,421,180 centurys, 18 years",
    "Huge cluster / botnet (~1T/s)": "11,746,586,421 centurys, 18 years"
  },
  "breach_check": {
    "breached": false,
    "count": 0
  }
}
```

---

### 2.2 `POST /improve`
- **Request Body:**
```json
{
  "password": "string"
}
```

- **Response Body (`200 OK`):**
```json
{
  "original": "password123",
  "suggestions": [
    {
      "strategy": "Memorable Passphrase",
      "password": "Password123!@#2025",
      "strength": "STRONG",
      "score": 1.0,
      "entropy": 117.98
    },
    {
      "strategy": "Smart Leetspeak",
      "password": "S@fePassword1237!",
      "strength": "STRONG",
      "score": 1.0,
      "entropy": 111.43
    },
    {
      "strategy": "Cryptographic Master",
      "password": "PASSword123#Safe!X",
      "strength": "STRONG",
      "score": 1.0,
      "entropy": 117.98
    }
  ]
}
```

---

### 2.3 `POST /export_report`
- **Request Body:**
```json
{
  "password": "string"
}
```

- **Response Body (`200 OK`):**
```json
{
  "filename": "PassShield_Audit_Report.md",
  "report": "# 🛡️ PassShield Security Audit Report\n..."
}
```

---

## 3. Resilience & Graceful Fallback Architecture

### 3.1 TensorFlow / Python 3.14 Compatibility Layer
- **Issue:** TensorFlow does not yet support Python 3.14 runtimes.
- **Solution:** `app.py` implements try/except import handling. If TensorFlow is unavailable, `_lstm_predict()` gracefully returns `None`, and the ensemble base score dynamically adjusts to average the active models (`SVM`, `RF`, `XGBoost`).

### 3.2 Offline / Groq API Failover
- **Issue:** Network disconnects or API quota limits could disrupt Groq LLM calls.
- **Solution:** `_fallback_suggestion()` steps in automatically to produce rule-governed high-entropy password candidates ensuring 100% operational uptime.

---

## 4. Frontend Design System Specs

- **Background:** Multi-layer radial gradients with CSS keyframe animation (`bgShift` 12s infinite).
- **Glassmorphism:** CSS backdrop filters (`backdrop-filter: blur(20px) saturate(180%)`) with semi-transparent borders (`rgba(255,255,255,0.1)`).
- **Typography:**
  - UI Labels & Headers: Google Font **Inter** (weights 300 to 800)
  - Passwords & Metrics: Google Font **JetBrains Mono**
- **Color Tokens:**
  - Primary Accent: `#6c63ff` (Violet) to `#00d4ff` (Cyan) linear gradient
  - Weak Status: `#ff4d6d` (Rose Red)
  - Medium Status: `#ffb347` (Amber)
  - Strong Status: `#3dfaaf` (Mint Green)
