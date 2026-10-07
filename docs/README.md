# 🛡️ PassShield — AI Password Intelligence Platform

Welcome to the official documentation directory for **PassShield**, a full-stack ML password security analyzer engineered for modern enterprise security and FAANG technical interviews.

---

## 📚 Documentation Index

| Guide | Description |
|---|---|
| 🏗️ **[Architecture Overview](ARCHITECTURE.md)** | High-level system architecture, dataflow diagrams, component breakdown, and tech stack choices. |
| 🧠 **[ML Ensemble & Math](WORKING_AND_ML_ENSEMBLE.md)** | Detailed mathematical & algorithmic explanations of the 4-Model Ensemble (SVM, Random Forest, XGBoost, LSTM), Shannon entropy math, pattern score adjustments, and time-to-crack formulas. |
| 🛡️ **[Security & Uniqueness](SECURITY_AND_UNIQUENESS.md)** | In-depth breakdown of Zero-Knowledge HaveIBeenPwned k-Anonymity breach detection, Groq LLM prompt strategies, and interview standout matrix. |
| ⚙️ **[System Design & API](SYSTEM_DESIGN.md)** | Technical SLA, complete API contracts (`/analyze`, `/improve`, `/export_report`), offline fallbacks, and design system tokens. |
| 🚀 **[Free Cloud Deployment Guide](DEPLOYMENT.md)** | Step-by-step instructions for 100% free deployment on Render, Hugging Face Spaces, or Railway. |

---

## 🚀 Quickstart Guide

Security deployment planning and the post-deployment evidence checklist are in [Security Deployment Plan](SECURITY_DEPLOYMENT_PLAN.md).

### 1. Installation
```powershell
cd f:\Project\password\password_analyzer
..\venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 2. Configure Environment
Create or edit `.env` in the `password_analyzer` folder:
```env
GROQ_API_KEY=gsk_your_groq_api_key_here
```

### 3. Run Application
```powershell
python app.py
```
Open **[http://localhost:5000](http://localhost:5000)** in your browser.

---

## 🎯 FAANG Technical Interview Q&A Cheat Sheet

### Q1: Why use a 4-Model ML Ensemble (SVM, RF, XGB, LSTM) instead of a single model?
> **Answer (Deep Technical):** A single classifier suffers from fundamental boundary limitations, high model variance, and adversarial blind spots. 
> 1. **Variance Reduction:** Heterogeneous ensemble math dictates that averaging uncorrelated algorithms ($\rho \to 0$) reduces generalization variance according to $\text{Var}(\bar{f}) = \rho\sigma^2 + \frac{1-\rho}{N}\sigma^2$.
> 2. **Boundary Coverage:** SVM fits maximum-margin hyperplanes in continuous space; Random Forest & XGBoost fit orthogonal step boundaries for discrete character interactions; and LSTM models Markovian character sequence transitions ($x_t \mid h_{t-1}$).
> 3. **Adversarial Defense:** A password like `P@ssw0rd2026!` tricks decision trees (passes length/symbols), but gets caught by LSTM's sequence memory.
> 4. **Epistemic Uncertainty:** Model probability variance ($\text{Var}(P_i)$) flags edge-case inputs for secondary heuristic verification.
> 5. **Fault Tolerance:** If a framework runtime (e.g. TensorFlow on Python 3.14) fails, the ensemble dynamically recalculates consensus without server downtime.

### Q2: How does your breach detection guarantee zero privacy leaks?
> **Answer:** PassShield uses the **k-Anonymity protocol**. The password is never sent over the network. Instead, the backend computes its SHA-1 hash locally, sends **only the first 5 hex characters** to the HaveIBeenPwned API, and receives a list of ~500-1000 candidate hash suffixes. PassShield then checks locally if the suffix matches. Because over a million passwords share the same 5-character prefix, neither HIBP nor any network observer can ever identify the password being checked.

### Q3: How do you calculate time-to-crack estimates?
> **Answer:** Time-to-crack is calculated using Shannon Entropy ($H = L \times \log_2(C)$). The keyspace size is $2^H$. Assuming an attacker brute-forces half the keyspace ($2^{H-1}$ guesses on average), we divide by the throughput of 4 threat scenarios: Online Throttled (10/s), Offline CPU (1M/s), Fast GPU Cluster (1B/s), and Distributed Botnet (1T/s).

### Q4: How does the system handle model failures or missing dependencies?
> **Answer:** PassShield is built with **graceful degradation**. For example, since TensorFlow does not yet support Python 3.14, the LSTM model returns `None`, and the ensemble dynamically re-weights across the active models (SVM, RF, XGBoost). If the Groq LLM API is unreachable, an automated rule-governed high-entropy generator steps in to guarantee 100% uptime.
