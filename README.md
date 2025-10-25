# EarlyDoc AI - An AI-Powered Disease Prediction & Diagnosis Assistant

## Overview

Early detection of diseases is crucial in preventing severe health complications. However, many individuals struggle with:
❌ Misinterpreting symptoms  
❌ Limited access to medical professionals  
❌ Slow self-research through unreliable sources  

To solve this, we present an **AI-powered Generative Health Diagnosis Assistant** that:
✅ Accepts user symptoms (Text/Voice)  
✅ Predicts possible diseases with confidence scores  
✅ Generates human-like explanations using GenAI  
✅ Suggests next steps (tests, urgency, doctor type)  
✅ Outputs structured JSON (API friendly)

---

## 🎯 Objective

Build a **conversational AI health assistant** powered by:
📊 Machine Learning (Symptom → Disease Prediction)  
🧠 LLM-based Explanation Generation  
🎙️ Voice Input + TTS Output  
📂 JSON-structured response for scalability  

---

## 🦠 Diseases Covered (Phase 1)

| Disease | Severity | Reason | Dataset |
|---------|----------|--------|--------|
| Diabetes | Medium | High prevalence | ✅ Available |
| Heart Disease | High | Critical & time-sensitive | ✅ Available |
| Pneumonia | High | Rapid escalation | ✅ Available |
| Malaria | Medium | Regionally significant | ✅ Available |
| COVID-19 / Flu | Medium | Symptom-overlap challenges | ✅ Available |
| Migraine | Low | Common misinterpretation | ✅ Available |

✅ (Optional) Phase 2: Depression/Anxiety classification

---

## 📂 Datasets Used

| Name | Use | Source |
|------|-----|--------|
| Disease Symptom Prediction CSV | Multi-disease classification | Kaggle |
| PIMA Diabetes | Diabetes probability | Kaggle |
| UCI Cleveland Heart Disease | Heart issues | UCI/Kaggle |
| Malaria & Pneumonia Symptoms CSV | Symptom mapping | Kaggle |
| COVID-19 vs Flu Data | Viral infections | Kaggle |

✅ All dataset links will be provided in `/datasets/README.md`

---

## 🤖 AI Models Used

| Component | Model |
|-----------|-------|
| Symptom → Disease | Random Forest / XGBoost |
| Confidence Boost | Logistic Regression |
| Symptoms Understanding | Llama 3 / Gemma-Instruct |
| Response Generation | LLM (Gemini-Pro / Llama 3) |
| Voice Input | Whisper / Vosk |
| Speech Output | gTTS / Coqui TTS |

---

## ⚙️ System Architecture (Workflow Overview)

```mermaid
graph TD;
A[🧑 User: Symptom Input (Text/Voice)] --> B[🎙️ Voice to Text (Whisper/Vosk)]
B --> C[🧠 NLP Symptom Extraction]
C --> D[📊 ML Model: Disease Prediction]
D --> E[📈 Ranked Results + Confidence Scores]
E --> F[🧠 GenAI: Explanation + Next Steps Generation]
F --> G[📤 JSON Response + UI Display]
