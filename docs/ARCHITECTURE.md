# 🏗️ PassShield — System Architecture

## 1. High-Level Architecture Overview

PassShield is engineered as a **hybrid multi-layered security application** combining classic heuristic entropy math, a 4-algorithm Machine Learning classification ensemble, zero-knowledge network privacy protocols, and Large Language Model (LLM) intelligence.

```
                  ┌──────────────────────────────────────────────┐
                  │            Client Browser (Frontend)         │
                  │  - Real-Time Debounced Analyzer (500ms)      │
                  │  - Canvas API Radial Strength Gauge          │
                  │  - Glassmorphism Reactive UI System          │
                  │  - Client-Side Wasm/Crypto Generator         │
                  └──────────────────────┬───────────────────────┘
                                         │  HTTP POST /analyze, /improve, /export_report
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │            Flask Web Application Server      │
                  │             (Python 3.14 Backend)            │
                  └──────┬──────────────────────┬─────────┬──────┘
                         │                      │         │
       ┌─────────────────┴────────┐             │         └──────────────────────────┐
       ▼                          ▼             ▼                                    ▼
┌──────────────┐         ┌───────────────────────────┐                    ┌────────────────────┐
│ Feature      │         │ 4-Model ML Ensemble Engine│                    │ Zero-Knowledge HIBP│
│ Extraction   │         ├───────────────────────────┤                    │ Breach Checker     │
│ & Entropy    │         │ 1. SVM (Support Vector)   │                    │ (k-Anonymity SHA1) │
│ Engine       │         │ 2. Random Forest (RF)     │                    └────────────────────┘
└──────────────┘         │ 3. XGBoost (Gradient Boost│
                         │ 4. LSTM (Sequence RNN)    │
                         └───────────────────────────┘
                                       │
                                       ▼
                         ┌───────────────────────────┐
                         │ Groq REST API (LLM)       │
                         │ DeepSeek R1 70B Distill   │
                         └───────────────────────────┘
```

---

## 2. Component Breakdown

### 2.1 Frontend Tier (`templates/index.html`, `static/style.css`, `static/script.js`)
- **Reactive Engine (`script.js`)**: Evaluates password input on a 500ms debounce loop. Communicates asynchronously via `fetch()` API.
- **Canvas Visualizer**: Rendered using HTML5 Canvas API with GPU acceleration (`requestAnimationFrame`) for 60fps radial strength gauges.
- **Design System (`style.css`)**: Dark mode glassmorphism UI leveraging CSS custom properties, backdrop filters (`backdrop-filter: blur(20px)`), and keyframe animations.
- **Client-Side Generator**: Uses `window.crypto.getRandomValues()` for cryptographically secure random number generation (CSPRNG).

### 2.2 API Service Layer (`app.py`)
- **`POST /analyze`**: Orchestrates feature extraction, ML model probability calculation, score attribution math, crack time estimation, and k-Anonymity breach detection. Returns JSON payload.
- **`POST /improve`**: Receives target password, constructs 3 distinct prompt strategies (*Memorable Passphrase*, *Smart Leetspeak*, *Cryptographic Master*), queries the Groq API via direct REST HTTP call, and evaluates candidates through the ML engine.
- **`POST /export_report`**: Generates a structured enterprise Security Audit Report in Markdown format.

### 2.3 Feature Extraction Engine (`utils/features.py`)
- Extracts a 6-dimensional numerical feature vector:
  1. `length`: Total character count
  2. `has_upper`: Binary indicator (0/1)
  3. `has_lower`: Binary indicator (0/1)
  4. `has_digit`: Binary indicator (0/1)
  5. `has_special`: Binary indicator (0/1)
  6. `entropy`: Shannon entropy calculation ($\text{bits} = L \times \log_2(\text{charset})$)

---

## 3. Data Flow Diagram

```
User types "P@ssw0rd2026"
  │
  ├──> [Client Debounce 500ms]
  │      └──> POST /analyze {"password": "P@ssw0rd2026"}
  │             │
  │             ├──> extract_features() ──> [12, 1, 1, 1, 1, 78.6]
  │             │
  │             ├──> ML Models Predict Proba:
  │             │      ├── SVM: 0.999
  │             │      ├── RF: 1.000
  │             │      ├── XGB: 0.998
  │             │      └── LSTM: N/A (Python 3.14 fallback)
  │             │
  │             ├──> pattern_score() ──> -0.25 (common word "password" detected)
  │             │
  │             ├──> check_pwned_api() ──> SHA1("P@ssw0rd2026") ──> Prefix "7C4A8"
  │             │      └──> GET https://api.pwnedpasswords.com/range/7C4A8
  │             │             └──> Response match: 1,420 breaches
  │             │
  │             └──> Construct JSON Response
  │                    └──> Client renders: Gauge (85%), Warning Banner, Model Bars
```
