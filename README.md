# Predictive ICU Monitor: AI-Powered Predictive ICU Monitoring System 🩺

Predictive ICU Monitor is an advanced clinical decision support system designed to combat **alarm fatigue** in Intensive Care Units (ICUs). By leveraging machine learning and generative AI, it **proactively detects** subtle patterns in patient vital signs to predict deterioration, specifically sustained hypotension, hours before it becomes critical. This allows for earlier intervention and aims to improve **patient safety**.

## 📖 Table of Contents

* [The Problem: Alarm Fatigue](#-the-problem-alarm-fatigue)
* [Our Solution: Two-Stage Predictive Monitoring](#-our-solution-two-stage-predictive-monitoring)
* [⚙️ Workflow](#️-workflow)
* [🚀 Technology Stack](#-technology-stack)
* [📁 Project Structure](#-project-structure)
* [🛠️ Setup & Installation](#️-setup--installation)
* [▶️ Running the Project](#️-running-the-project)
* [🖱️ How to Use the Dashboard](#️-how-to-use-the-dashboard)
* [💡 Future Enhancements](#-future-enhancements)

## 😟 The Problem: Alarm Fatigue

Current ICU monitoring often relies on static threshold alarms (e.g., alert if Heart Rate > 120 bpm). This system suffers from several drawbacks:

1.  **Reactive:** Alarms typically trigger *after* a patient's condition has already significantly worsened.
2.  **Univariate:** It checks each vital sign in isolation, often missing subtle, multivariate patterns that indicate early decline (e.g., a slow, simultaneous drop in Mean Arterial Pressure (MAP) combined with a rise in Heart Rate (HR)).
3.  **Noisy:** High rates of false or non-actionable alarms lead to **alarm fatigue**, where clinical staff become desensitized and may delay responses to genuine critical events.

---

## ✨ Our Solution: Two-Stage Predictive Monitoring

Predictive ICU Monitor employs a **two-stage pipeline** to transform raw vital sign data into actionable clinical insights:

1.  **Stage 1: Machine Learning Engine (Pattern Detection)**
    * **Goal:** Predict the onset of sustained hypotension (MAP < 65 mmHg for 10+ minutes) within the next 120 minutes.
    * **Method:** An **XGBoost Classifier** analyzes engineered features (trends, volatility, means) calculated over a 30-minute sliding window of vital signs (HR, MAP, RR, SpO2).
    * **Output:** A structured JSON object containing patient details, the predicted risk score, and key data points triggering the prediction.

2.  **Stage 2: Generative AI Layer (Clinical Interpretation)**
    * **Goal:** Translate the technical JSON alert into a concise, actionable clinical assessment for the nurse.
    * **Method:** The JSON data is sent to a powerful Large Language Model (**Google Gemini**).
    * **Output:** A natural language summary assessing the primary concern, key evidence, and recommended next steps, displayed directly in the user interface.

    ## 🛠️ Setup & Installation

1.  **Clone the Repository:**
    ```bash
    git clone <your-repository-url>
    cd <your-project-folder-name> # e.g., cd Group_BC
    ```

2.  **Create and Activate a Virtual Environment:**
    ```bash
    # Windows
    python -m venv venv
    .\venv\Scripts\activate

    # macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install Dependencies:**
    * Ensure you have a `requirements.txt` file. If not, create one in your activated environment: `pip freeze > requirements.txt`
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up Gemini API Key:**
    * Visit [Google AI Studio](https://aistudio.google.com/app/apikey) to get your free API key.
    * Create a file named `.env` in the project root directory (same level as `app.py`).
    * Add the following line to the `.env` file, replacing `YOUR_API_KEY_HERE` with your actual key:
        ```
        GEMINI_API_KEY=YOUR_API_KEY_HERE
        ```
    * **Important:** Add `.env` to your `.gitignore` file to avoid accidentally committing your API key.

5.  **Ensure Model and Scaler Exist:**
    * Make sure the `feature_scaler.pkl` and `xgboost_stage1_model.pkl` files are present in the `/model` directory. If not, you may need to run the `train_xgboost.py` script first (using appropriate training data).

---

## ▶️ Running the Project

1.  **Activate your virtual environment** (if not already active):
    ```bash
    # Windows: .\venv\Scripts\activate
    # macOS/Linux: source venv/bin/activate
    ```
2.  **Start the Flask Server:**
    ```bash
    python app.py
    ```
3.  **Access the Dashboard:** Open your web browser and navigate to `http://127.0.0.1:5000` (or the address provided in the terminal).

---

## 🖱️ How to Use the Dashboard

1.  **Start Simulation:** Click the "Start Simulation" button to begin processing the patient data stream.
2.  **Monitor Vitals:** Watch the real-time graphs for HR, MAP, RR, and SpO2. The graphs show the most recent 30 data points and scroll automatically.
3.  **Review Alerts:** When the system predicts a high risk:
    * The corresponding data point on the graph will turn **red**.
    * An alert card will appear below the graphs, displaying the **LLM's clinical assessment**.
    * A pulsing **"SNOOZE ⏰"** button will appear in the top status bar.
4.  **Acknowledge Alert:** Read the LLM assessment on the alert card and click the **"Mark as Read"** button. The button will change to "Acknowledged ✓".
5.  **Snooze Alarm:** *After* marking the alert as read, you can click the "SNOOZE ⏰" button in the status bar. This will silence further alerts *for the current high-risk period*. The snooze automatically resets when the patient's predicted risk drops below the threshold.
    * If you try to click Snooze before Marking as Read, you'll get a prompt.
6.  **Stop Simulation:** Click the "Stop Simulation" button at any time to end the process.

# 🏥 Patient Deterioration Early Warning System

An AI-powered real-time patient monitoring solution that predicts deterioration risks using XGBoost and augments predictions with natural language assessments from Google Gemini. The results are delivered dynamically on a live dashboard built using Flask, SocketIO, and Chart.js.

---

## ⚙️ Workflow Overview

The system processes, predicts, and presents insights through the following pipeline:

1. 📥 **Input CSV** containing patient vital signs (e.g., heart rate, blood pressure).
2. 📈 **Feature Extraction** over a 30-minute sliding window (trends, averages, statistical metrics).
3. 📏 **Normalization** using `StandardScaler` to scale features.
4. 🤖 **Prediction** via XGBoost to calculate probability of patient deterioration.
5. 🚨 **Threshold Evaluation** (e.g., > 49% risk deemed critical).
6. 📦 **JSON Generation** including prediction score, patient info, and vital trends.
7. 🧠 **Google Gemini Analysis** of the JSON for human-like clinical interpretation.
8. 📡 **Live Visualization** of prediction & AI-generated insights via Flask + SocketIO + Chart.js.

---

## 🚀 Technology Stack

| Layer | Tools & Libraries |
|-------|------------------|
| **Backend** | Python, Flask, Flask-SocketIO, Eventlet |
| **Machine Learning** | XGBoost, Scikit-learn, Pandas, NumPy, Joblib |
| **Generative AI** | Google Gemini (`google-generativeai`) |
| **Frontend** | HTML, CSS, JavaScript, Chart.js |
| **Utilities** | `python-dotenv`, `venv` |

---

## 📁 Project Structure

├── data/ # CSV files for vitals (raw/simulation/training)
├── model/ # Trained ML model & scaler
│ ├── xgboost_stage1_model.pkl
│ └── feature_scaler.pkl
├── templates/ # Frontend HTML templates
│ └── index.html
├── app.py # Runs ML inference, LLM analysis, dashboard UI
├── train_xgboost.py # Training script for XGBoost model
├── requirements.txt # Dependencies
├── .env # Gemini API key (excluded from Git)
└── README.md # Documentation

## ⚙️ Workflow – How the System Works

This section explains the step-by-step flow of the Patient Deterioration Early Warning System, from raw vital signs to AI-generated clinical insights displayed on the dashboard.

---

### 🔍 1. Input: Patient Vital Signs
- The system reads raw physiological data (e.g., heart rate, blood pressure, respiratory rate, oxygen saturation) from a CSV file located in the `/data` directory.
- Each row represents readings at a specific timestamp.

---

### 📈 2. Feature Engineering (30-Min Sliding Window)
- A 30-minute sliding window is used to calculate statistical and trend-based features such as:
  ✅ Moving average  
  ✅ Rate of change  
  ✅ Standard deviation  
  ✅ Min/Max shifts  
- These engineered features help the model better understand patterns over time rather than single-point values.

---

### 📏 3. Feature Scaling
- All extracted features are normalized using a `StandardScaler` to maintain a consistent range.
- This avoids bias towards features with larger numerical values.

---

### 🤖 4. Machine Learning Prediction with XGBoost
- The pre-trained XGBoost model (`xgboost_stage1_model.pkl`) processes the scaled features.
- It outputs a probability score indicating the likelihood of patient deterioration.

---

### 🚨 5. Threshold Evaluation
- If the risk probability exceeds a chosen threshold (e.g., **49%**), the system classifies the situation as **high risk**.
- This triggers the generation of a detailed clinical interpretation request.

---

### 📦 6. JSON Generation
- The system compiles a structured JSON object that contains:
  ```json
  {
    "patient_id": "...",
    "current_vitals": {...},
    "feature_trends": {...},
    "risk_probability": 0.73,
    "status": "High Risk"
  }

## 🧠 7. Generative AI Analysis (Google Gemini)
The generated JSON is sent to the Google Gemini API using the google-generativeai Python library.

Gemini converts the structured data into a human-friendly clinical summary, explaining:
✅ Severity
✅ Possible causes
✅ Suggested escalation steps

## 📡 8. Real-Time Visualization (Flask + SocketIO)

The final output (prediction + AI assessment) is pushed in real-time to the web dashboard.

The frontend updates dynamically and displays:
✅ Live vital sign graphs (Chart.js)
✅ Deterioration risk score
✅ Natural language diagnosis summary
✅ Alert indicators for high-risk cases

## ✅ Final Outcome

-This workflow ensures clinicians receive data-driven risk predictions backed by AI-generated clinical reasoning, all presented on a real-time monitoring dashboard.

## Dataset Link
-https://vitaldb.net/dataset/

## 💡 Future Enhancements

* **RAG Integration:** Implement Retrieval-Augmented Generation (Stage 2) by connecting the LLM to a knowledge base of hospital protocols for more specific recommendations.
* **Real Data:** Train and validate the model on real, de-identified ICU patient data.
* **Model Retraining Pipeline:** Set up a system for periodically retraining the XGBoost model.
* **UI Improvements:** Add more detailed patient information, historical alert logs, and configuration options (like setting the threshold).
* **Deployment:** Package the application for deployment (e.g., using Docker).
* **Error Handling:** More robust error handling for API calls and data processing.
