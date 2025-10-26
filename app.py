import eventlet
eventlet.monkey_patch() # MUST BE THE VERY FIRST THING

import os
import json
import time
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from sklearn.preprocessing import StandardScaler
from flask import Flask, render_template
from flask_socketio import SocketIO
from dotenv import load_dotenv # New
import google.generativeai as genai # New

# --- 1. LOAD API KEY AND CONFIGURE LLM ---
load_dotenv()
API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    print("="*80)
    print("WARNING: GEMINI_API_KEY not found in .env file.")
    print("LLM analysis will be disabled. Please create a .env file.")
    print("="*80)
else:
    try:
        genai.configure(api_key=API_KEY)
        llm_model = genai.GenerativeModel('gemini-pro-latest')
        print("✓ Gemini LLM Model loaded successfully.")
    except Exception as e:
        print(f"ERROR: Could not configure Gemini. Check API key. Error: {e}")

# --- 2. INITIALIZE APP AND SOCKETIO ---
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key!'
socketio = SocketIO(app, async_mode='eventlet')

# --- Global state variables ---
simulation_running = False
snooze_is_active = False

# --- 3. CONFIGURATION & FILE PATHS ---
try:
    parent_dir_context = os.path.dirname(os.path.abspath(__file__))
except NameError:
     parent_dir_context = os.getcwd()

TEST_DATA_FILE = 'test_patient_data.csv'
MODEL_FILE = 'xgboost_stage1_model.pkl'
SCALER_FILE = 'feature_scaler.pkl'

data_dir = os.path.join(parent_dir_context, "data")
model_dir = os.path.join(parent_dir_context, "model")

TEST_DATA_PATH = os.path.join(data_dir, TEST_DATA_FILE)
MODEL_PATH = os.path.join(model_dir, MODEL_FILE)
SCALER_PATH = os.path.join(model_dir, SCALER_FILE)

VITAL_COLUMNS = ['HR', 'RR', 'SpO2', 'MAP']
WINDOW_SIZE = 30
OPTIMAL_THRESHOLD = 0.49 # Your threshold

# --- 4. HELPER FUNCTIONS ---
def calculate_features(window_df):
    """Calculates features for a single window."""
    if len(window_df) < WINDOW_SIZE: return {}
    mean_hr = window_df['HR'].mean()
    median_hr = window_df['HR'].median()
    mean_map = window_df['MAP'].mean()
    median_map = window_df['MAP'].median()
    last_hr = window_df['HR'].iloc[-1]
    last_map = window_df['MAP'].iloc[-1]
    first_hr = window_df['HR'].iloc[0]
    features = {}
    for col in VITAL_COLUMNS:
        features[f'{col}_mean'] = window_df[col].mean()
        features[f'{col}_median'] = window_df[col].median()
        features[f'{col}_std'] = window_df[col].std()
        features[f'{col}_min'] = window_df[col].min()
        features[f'{col}_max'] = window_df[col].max()
        features[f'{col}_trend'] = last_hr - first_hr
    features['HR_diff_from_mean'] = last_hr - mean_hr
    features['HR_diff_from_median'] = last_hr - median_hr
    features['MAP_diff_from_mean'] = last_map - mean_map
    features['MAP_diff_from_median'] = last_map - median_map
    features['HR_MAP_product_mean'] = (window_df['HR'] * window_df['MAP']).mean()
    features['SpO2_RR_ratio_mean'] = (window_df['SpO2'] / window_df['RR']).mean()
    features['SpO2_RR_ratio_min'] = (window_df['SpO2'] / window_df['RR']).min()
    features['HR_MAP_product_mean.1'] = features['HR_MAP_product_mean']
    features['HR_diff_from_mean.1'] = features['HR_diff_from_mean']
    features['HR_diff_from_median.1'] = features['HR_diff_from_median']
    features['MAP_diff_from_mean.1'] = features['MAP_diff_from_mean']
    features['MAP_diff_from_median.1'] = features['MAP_diff_from_median']
    return features

def create_json_alert(patient_id, window_end_time, raw_data_window, prediction_proba):
    """Generates the comprehensive JSON output for Stage 1."""
    last_row = raw_data_window.iloc[-1]
    first_row = raw_data_window.iloc[0]
    hr_trend_display = last_row['HR'] - first_row['HR']
    map_trend_display = last_row['MAP'] - first_row['MAP']
    rr_trend_display = last_row['RR'] - first_row['RR']
    if map_trend_display < -5 and hr_trend_display > 10 and last_row['MAP'] < 70:
        pattern_code = "EARLY_SHOCK_P-A"
        pattern_desc = "Compensated shock: MAP falling while HR rises."
    elif rr_trend_display > 4 and last_row['SpO2'] < 96:
        pattern_code = "RESP_DISTRESS_P-B"
        pattern_desc = "Respiratory distress: RR rising with low SpO2."
    else:
        pattern_code = "RISK_HIGH_GENERIC"
        pattern_desc = "High-Risk Pattern Detected (Non-Specific)"
    json_output = {
        "patient_id": str(patient_id), "window_end_time": str(window_end_time),
        "pattern_detected": str(pattern_code), "pattern_description": str(pattern_desc),
        "risk_score_proba": float(round(prediction_proba, 4)),
        "analysis_window": f"{WINDOW_SIZE} minutes",
        "demographics": { "age": int(last_row['age']), "sex": str(last_row['sex']), "bmi": float(round(last_row['bmi'], 1)) },
        "trigger_data_trends": { "current_HR": float(round(last_row['HR'], 1)), "current_MAP": float(round(last_row['MAP'], 1)), "current_RR": float(round(last_row['RR'], 1)), "HR_trend": f"{hr_trend_display:.1f} bpm", "MAP_trend": f"{map_trend_display:.1f} mmHg", },
        "raw_window_summary": { "min_HR": float(round(raw_data_window['HR'].min(), 1)), "min_SpO2": float(round(raw_data_window['SpO2'].min(), 1)), "avg_MAP": float(round(raw_data_window['MAP'].mean(), 1)) }
    }
    return json_output

# --- UPDATED: LLM Analysis Function with Better Error Handling ---
def get_llm_analysis(alert_json):
    """Sends the alert JSON to the LLM and gets an analysis."""
    if llm_model is None:
        return "LLM analysis disabled. No API key found or configuration failed."

    prompt = f"""
    You are an expert ICU clinical assistant.
    You have received the following JSON data from a patient monitoring system.
    The system has detected a high-risk pattern.

    JSON DATA:
    {json.dumps(alert_json, indent=2)}

    Analyze this data and provide a brief, actionable assessment for the attending nurse.
    1. What is your primary concern (e.g., compensated shock, respiratory distress, potential sepsis, or false alarm)?
    2. What is the key evidence (e.g., "MAP is falling while HR is rising")?
    3. What is the recommended next step?

    Keep your response 3-4 sentences long and start with "Assessment:".
    """
    try:
        print("Sending request to LLM...") # Added print statement
        response = llm_model.generate_content(prompt)
        print("Received response from LLM.") # Added print statement
        # Check if the response has text (sometimes it might return empty or blocked)
        if hasattr(response, 'text') and response.text:
            return response.text
        else:
            # Try accessing parts if text is missing (useful for debugging blocked content)
            try:
                # Check for safety ratings if available
                if response.prompt_feedback.block_reason:
                    reason = response.prompt_feedback.block_reason
                    print(f"LLM Error: Request blocked due to {reason}")
                    return f"Error: LLM request blocked ({reason}). The prompt might contain sensitive content."
            except (AttributeError, IndexError):
                 # If no block reason, it might just be an empty response
                 pass # Fall through to generic error

            print("LLM Error: Received empty response or response missing 'text' attribute.")
            return "Error: Received an empty or malformed response from the LLM."

    except Exception as e:
        # Print the specific error from the API call
        print(f"LLM API Error: {type(e).__name__}: {e}")
        return f"Error: Could not get LLM analysis. API Error: {type(e).__name__}"

# --- 5. LOAD MODELS ON STARTUP ---
print("Loading model and scaler... This may take a moment.")
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_cols_model = model.feature_names_in_.tolist()
    df_sim = pd.read_csv(TEST_DATA_PATH)
    df_sim['time_dt'] = pd.to_datetime(df_sim['time'], format='%H:%M:%S', errors='coerce')
    df_sim = df_sim.dropna(subset=['time_dt'])
    print(f"✓ Model, scaler, and test data loaded successfully for {len(df_sim)} rows.")
except Exception as e:
    print(f"FATAL ERROR: Could not load model/scaler/data: {e}")
    model = None

# --- 6. DEFINE WEB ROUTES ---

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    global snooze_is_active
    snooze_is_active = False
    print('Client connected')

@socketio.on('stop_simulation')
def handle_stop_simulation():
    global simulation_running, snooze_is_active
    simulation_running = False
    snooze_is_active = False
    print("Simulation stop requested by client.")

@socketio.on('snooze_clicked')
def handle_snooze():
    """Called when the user clicks the 'Snooze' button."""
    global snooze_is_active
    snooze_is_active = True
    print("Alerts snoozed by user.")
    # Tell the frontend to hide the button again
    socketio.emit('deactivate_alarm')

@socketio.on('start_simulation')
def handle_start_simulation():
    """Runs the simulation loop in a background thread."""
    global simulation_running, snooze_is_active
    if simulation_running:
        print("Simulation already running.")
        return

    simulation_running = True
    snooze_is_active = False # Reset snooze on start

    print("Simulation requested... Starting loop.")
    socketio.emit('status_update', {'msg': 'Simulation started... Monitoring patient data.'})

    triggered_alerts = 0
    patient_id = df_sim['patient_id'].iloc[0] if not df_sim.empty else "P-SIM-000"

    for end_index in range(len(df_sim)):
        if not simulation_running:
            print("Simulation loop terminated.")
            socketio.emit('status_update', {'msg': 'Simulation Stopped by User.'})
            socketio.emit('deactivate_alarm') # Hide snooze button
            socketio.emit('simulation_ended')
            break

        if end_index < WINDOW_SIZE - 1:
            continue

        start_index = end_index - WINDOW_SIZE + 1
        window_data = df_sim.iloc[start_index : end_index + 1].copy()
        current_vitals = window_data.iloc[-1]
        window_end_time = current_vitals['time']

        vitals_data = {
            'time': window_end_time,
            'hr': float(current_vitals['HR']), 'map': float(current_vitals['MAP']),
            'rr': float(current_vitals['RR']), 'spo2': float(current_vitals['SpO2'])
        }
        socketio.emit('simulation_tick', vitals_data)

        # --- Feature Engineering ---
        features_dict = calculate_features(window_data)
        X_live = pd.DataFrame([features_dict])
        X_live['age'] = window_data['age'].iloc[-1]; X_live['bmi'] = window_data['bmi'].iloc[-1]
        X_live['sex_encoded'] = 1 if window_data['sex'].iloc[-1] == 'M' else 0
        if 'sex' in X_live.columns: X_live = X_live.drop(columns=['sex'])
        X_live = X_live.reindex(columns=feature_cols_model, fill_value=0.0)

        try:
            X_scaled = scaler.transform(X_live)
        except Exception as e:
            print(f"Scaling failed at {window_end_time}: {e}")
            continue

        # --- Prediction ---
        prediction_proba = model.predict_proba(X_scaled)[:, 1][0]

        # --- MODIFIED: LLM INTEGRATION LOGIC ---
        if prediction_proba >= OPTIMAL_THRESHOLD:
            if not snooze_is_active:
                print(f"ALERT: Triggered at {window_end_time} (P: {prediction_proba:.4f})")
                triggered_alerts += 1

                # 1. Generate the JSON
                alert_json = create_json_alert(
                    patient_id, window_end_time, window_data, prediction_proba
                )

                # 2. Get LLM Analysis
                llm_response = get_llm_analysis(alert_json)

                # 3. Create new payload
                new_payload = {
                    'alert_data': alert_json,
                    'llm_analysis': llm_response
                }

                # 4. Emit the new event
                socketio.emit('new_llm_alert', new_payload)
                socketio.emit('activate_alarm')
        else:
            if snooze_is_active:
                print("Patient risk dropped. Resetting snooze.")
            snooze_is_active = False
            socketio.emit('deactivate_alarm')
        # --- END OF MODIFICATION ---

        # --- SLOWED DOWN SIMULATION SPEED ---
        socketio.sleep(1.0) # 1000ms (1 second) per row

    if simulation_running:
        print(f"Simulation complete. Total alerts: {triggered_alerts}")
        socketio.emit('status_update', {'msg': f'Simulation Finished. Total alerts: {triggered_alerts}'})
        socketio.emit('deactivate_alarm') # Hide snooze button
        socketio.emit('simulation_ended')
        simulation_running = False

# --- 7. RUN THE WEB SERVER ---
if __name__ == '__main__':
    print("Starting Flask server... Open http://127.0.0.1:5000 in your browser.")
    socketio.run(app, host='0.0.0.0', port=5000)