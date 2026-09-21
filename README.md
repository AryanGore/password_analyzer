# 🔐 PassShield — AI-Powered Password Strength Analyzer

PassShield is an AI-powered password security application that combines **Machine Learning, Deep Learning, Entropy Analysis, and Generative AI** to analyze password strength and provide intelligent password improvement suggestions.

The system uses a **4-model ML ensemble** consisting of:

- Support Vector Machine (SVM)
- Random Forest
- XGBoost
- LSTM

It also provides entropy-based analysis, model-wise predictions, an overall strength score, explainability, and optional GenAI-powered password improvement using the Groq API.

---

## 🌐 Live Demo

🚀 **Try PassShield Online:**

👉 https://passshield-analyzer.onrender.com/

---

## 📦 GitHub Repository

👉 https://github.com/AryanGore/password_analyzer

---

# 🚀 Features

- 🔎 Password strength analysis
- 🤖 4-model machine learning ensemble
- 🧠 Support Vector Machine (SVM)
- 🌲 Random Forest
- ⚡ XGBoost
- 🔄 LSTM character-sequence analysis
- 🧮 Entropy-based analysis
- 📊 Model-wise prediction visualization
- 📈 Final password strength score
- 🏷️ Password classification
  - WEAK
  - MEDIUM
  - STRONG
- 🔍 Explainable evaluation trace
- ✨ AI-powered password improvement suggestions
- 🔗 Groq LLM integration
- 🌐 Flask REST API
- 💻 Interactive web frontend
- 🧪 Model testing utilities
- ☁️ Render deployment

---

# 🧠 Project Overview

Traditional password-strength checkers generally rely on fixed rules such as:

```text
Length > 8
Has uppercase


Has lowercase
Has number
Has special character

The system evaluates a password through multiple stages:
                    User Password
                         │
                         ▼
                 Feature Extraction
                         │
          ┌──────────────┼──────────────┐
          │              │              │
          ▼              ▼              ▼
         SVM       Random Forest      XGBoost
          │              │              │
          └──────────────┼──────────────┘
                         │
                         ▼
                        LSTM
                         │
                         ▼
                 Ensemble Evaluation
                         │
                         ▼
                Entropy + Score Logic
                         │
                         ▼
             Strength Classification
                  ┌──────┼──────┐
                  ▼      ▼      ▼
                WEAK   MEDIUM  STRONG
                         │
                         ▼
                   Explainability
                         │
                         ▼
                Optional GenAI Layer
                         │
                         ▼
                      Groq LLM

System Architecture :-
┌───────────────────────────┐
│       User / Browser      │
└─────────────┬─────────────┘
              │
              ▼
┌───────────────────────────┐
│    HTML / CSS / JS        │
│        Frontend           │
└─────────────┬─────────────┘
              │
              ▼
┌───────────────────────────┐
│        Flask API          │
├───────────────────────────┤
│ GET  /                    │
│ POST /analyze             │
│ POST /improve             │
└──────────┬─────────┬──────┘
           │         │
           │         └─────────────────┐
           │                           │
           │                           ▼
           │                  ┌─────────────────┐
           │                  │    Groq LLM     │
           │                  │  GenAI Layer    │
           │                  └─────────────────┘
           │
           ▼
┌───────────────────────────┐
│    Feature Extraction     │
├───────────────────────────┤
│ • Length                  │
│ • Uppercase               │
│ • Lowercase               │
│ • Digits                  │
│ • Special Characters      │
│ • Entropy                 │
└─────────────┬─────────────┘
              │
       ┌──────┼──────┬──────┐
       ▼      ▼      ▼      ▼
      SVM     RF   XGBoost  LSTM
       │      │      │      │
       └──────┼──────┴──────┘
              │
              ▼
┌───────────────────────────┐
│ Ensemble + Score Engine   │
├───────────────────────────┤
│ • Model predictions       │
│ • Pattern adjustment      │
│ • Entropy bonus           │
│ • Digit bonus             │
│ • Special character bonus │
└─────────────┬─────────────┘
              │
              ▼
┌───────────────────────────┐
│ Strength Classification   │
│                           │
│ WEAK / MEDIUM / STRONG    │
└─────────────┬─────────────┘
              │
              ▼
┌───────────────────────────┐
│ Explainability / JSON     │
└───────────────────────────┘
