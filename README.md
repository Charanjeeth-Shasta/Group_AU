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

| Disease        | Reason                     | Dataset     |
| -------------- | -------------------------- | ----------- |
| Diabetes       | High prevalence            | ✅ Available |
| Heart Disease  | Critical & time-sensitive  | ✅ Available |
| Pneumonia      | Rapid escalation           | ✅ Available |
| COVID-19 / Flu | Symptom-overlap challenges | ✅ Available |

✅ (Optional) Phase 2: Depression/Anxiety classification

---

| Name                                | Use                                                | Source                                                                         |
| ----------------------------------- | -------------------------------------------------- | ------------------------------------------------------------------------------ |
| PIMA Diabetes Dataset               | Predict diabetes probability from patient features | [Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) |
| UCI Cleveland Heart Disease Dataset | Predict heart disease risk from patient data       | [UCI/Kaggle](https://www.kaggle.com/datasets/ronitf/heart-disease-uci)         |
| Pneumonia Symptoms Dataset          | Symptom mapping for pneumonia                      | [Kaggle](https://www.kaggle.com/datasets) (choose small public subset)         |
| COVID-19 / Flu Symptoms Dataset     | Symptom mapping & viral infection prediction       | [Kaggle](https://www.kaggle.com/datasets) (COVID-19 & flu combined subset)     |


✅ All dataset links will be provided in `/datasets/README.md`

---

| Component                                | Model / Approach                                                                             | Guardrails & Evaluation                                                                                                 |
| ---------------------------------------- | -------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| Symptom → Disease Prediction             | Random Forest / XGBoost (Tabular datasets: Diabetes, Heart Disease, Pneumonia, COVID-19/Flu) | ✅ Feature validation, handle missing values, prevent unrealistic inputs                                                 |
| Symptom Interpretation / Understanding   | LLM (Llama 3 / Gemma-Instruct)                                                               | ✅ Prompt guardrails to avoid hallucinations; validation with real symptom data                                          |
| Generative Explanation & Recommendations | LLM (Llama 3) + **Guardrails**                                                               | ✅ Output filtered through Guardrails to prevent harmful advice; evaluation using test cases / expert-approved templates |
| Voice Input (Optional)                   | Whisper / Vosk                                                                               | ✅ Audio preprocessing checks (noise, length)                                                                            |
| Speech Output (Optional)                 | gTTS / Coqui TTS                                                                             | ✅ Ensure clarity & appropriate phrasing                                                                                 |
| Overall System Evaluation                | –                                                                                            | ✅ Accuracy metrics (Precision/Recall), ✅ Confusion Matrix, ✅ Risk-based test cases, ✅ User-safety validation            |


---

## ⚙️ System Architecture (Workflow Overview)

```mermaid
graph TD;
    A[User Input (Text/Voice)] --> B[Voice to Text (Whisper/Vosk)];
    B --> C[Symptom Extraction using NLP/LLM];
    C --> D[Disease Prediction Model (ML/XGBoost)];
    D --> E[Ranked Diseases with Confidence Scores];
    E --> F[GenAI Explanation + Next Steps];
    F --> G[JSON Response + UI Chat Display];
    G --> H[Optional Text-to-Speech Output];
