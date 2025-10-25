# EarlyDoc AI - An AI-Powered Disease Prediction & Diagnosis Assistant

A prototype predictive alert system that transforms noisy ICU data into proactive, actionable clinical insights using time-series analysis and Generative AI.

---

## 🔴 The Problem: Alarm Fatigue

In a modern ICU, the problem isn't a lack of data; it's **alarm fatigue**. Current systems are:

* **Noisy (High False Positives):** A patient's heart rate spike from a simple cough triggers the same alarm as a cardiac event. Nurses become desensitized and may miss a real crisis.
* **Late (Reactive):** Alarms only trigger *after* a vital sign has breached a critical threshold (e.g., SpO2 is already < 88%). The crisis is already in progress.
* **"Dumb" (Univariate):** They look at each vital sign in isolation. A skilled nurse sees a rising heart rate, rising respiratory rate, and falling blood pressure together as an early sign of sepsis. The system cannot.

## 💡 Our Solution: The Two-Stage "Smart" Pipeline

This project moves from reactive alarms to **proactive, predictive alerts**. It combines a time-series model with a RAG-powered GenAI layer.

### 🧠 Part 1: The ML Engine (The "Pattern Detector")

This component analyzes a "sliding window" of patient data (e.g., the last 30 minutes) to find subtle, multivariate patterns that are known to precede a crisis.

* **Input:** A (simulated) real-time stream of vitals (HR, RR, SpO2, MAP, Temp).
* **Process:** It doesn't just check the latest value. It analyzes the **trend and relationship** between vitals.
* **Output:** When it detects a high-risk pattern (like early-stage sepsis or respiratory distress), it outputs a structured JSON alert.

### 🤖 Part 2: The GenAI Layer (The "Clinical Interpreter")

The JSON alert from the ML engine is still "code." This GenAI layer translates that data into a human-readable, clinically relevant summary using **Retrieval-Augmented Generation (RAG)**.

* **Input:** The JSON alert from Part 1 (e.g., `{"pattern": "B_RESP_DISTRESS", ...}`).
* **Process (RAG):**
    1.  **Retrieve:** The system uses the pattern to find the hospital's specific written protocol for that event (e.g., retrieves `respiratory_distress_protocol.txt`).
    2.  **Augment:** It combines the real-time patient data (the JSON) with the retrieved protocol (the context).
    3.  **Generate:** It feeds this rich prompt into an LLM (like Google's Gemini) to generate a concise, trustworthy SBAR note.
* **Output:** An actionable alert for the nurse's dashboard.

---

## 🚀 Real-Time Walkthrough

1.  **Data Stream:** A (simulated) patient's vitals are streaming in. All are technically "within normal limits," but the trend is worsening.
    * `11:45 AM: HR: 98, RR: 22, SpO2: 96%` (No static alarm)
    * `12:00 PM: HR: 110, RR: 26, SpO2: 93%` (Still no static alarm!)

2.  **ML Engine Fires:** The ML model analyzes this 15-minute window, flags a match for **Pattern B: Respiratory Distress**, and sends its JSON alert.

3.  **GenAI Layer Translates:** The LLM receives the JSON and retrieves the hospital's protocol. It instantly generates the SBAR note.

4.  **Actionable Alert:** The nurse's dashboard, instead of a simple "HR HIGH" alarm, shows this:

> **ALERT: Patient P-451 (Room 204B) - Potential Respiratory Distress**
>
> **Situation:** Patient is showing a 15-minute deteriorating trend consistent with early respiratory distress.
> **Background:** Vitals have trended from HR 90 -> 110 bpm, Respiration 18 -> 26 breaths/min, and SpO2 97% -> 93%.
> **Assessment:** This combined pattern indicates a high risk of an impending crisis, even though no single vital has breached a critical limit.
> **Recommendation (Per Protocol 7.2):**
> * Immediate tableside assessment.
> * Order STAT arterial blood gas (ABG).
> * Notify on-call respiratory therapist.

This alert is **proactive** (predicts the crisis), **high-fidelity** (it's not a false alarm), and **actionable** (it tells the nurse exactly what to do).

---

## 🛠️ Tech Stack

* **Frontend & Backend:** Streamlit
* **GenAI / LLM:** Google Gemini API
* **Simulation & Logic:** Python
* **Data Handling:** Pandas

---

## 📂 Project Structure

```bash
.
├── 📄 app.py                   # The main Streamlit dashboard application
├── 📄 genai_interpreter.py      # Contains all logic for calling the LLM and RAG
├── 📄 mock_ml_engine.py          # A "mock" function that simulates the ML model
├── 📄 patient_sim.csv          # The simulated patient data feed
├── 📁 protocols/
│   ├── 📄 B_RESP_DISTRESS.txt    # RAG knowledge base for respiratory distress
│   └── 📄 A_SEPSIS.txt           # RAG knowledge base for sepsis
└── 📄 requirements.txt         # All Python dependencies
