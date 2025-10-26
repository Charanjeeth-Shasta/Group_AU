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

# --- 1. INITIALIZE APP AND SOCKETIO ---
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key!' 
socketio = SocketIO(app, async_mode='eventlet') 

# --- Global state variables ---
simulation_running = False
snooze_is_active = False # NEW: For manual snooze button

# --- 2. CONFIGURATION & FILE PATHS ---
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

# --- 3. HELPER FUNCTIONS (No Changes) ---
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

# --- 4. LOAD MODELS ON STARTUP (No Changes) ---
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

# --- 5. DEFINE WEB ROUTES ---

@app.route('/')
def index():
    return render_template('index.html') 

@socketio.on('connect')
def handle_connect():
    global snooze_is_active
    snooze_is_active = False # Reset snooze on new connection
    print('Client connected')

@socketio.on('stop_simulation')
def handle_stop_simulation():
    global simulation_running, snooze_is_active
    simulation_running = False
    snooze_is_active = False # Reset snooze on stop
    print("Simulation stop requested by client.")

# --- NEW: Handler for the Snooze Button ---
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
        
        # --- START OF UPGRADED SNOOZE LOGIC ---
        if prediction_proba >= OPTIMAL_THRESHOLD:
            # High risk detected. Check if snoozed.
            if not snooze_is_active:
                # Not snoozed. Fire the alert.
                print(f"ALERT: Triggered at {window_end_time} (P: {prediction_proba:.4f})")
                triggered_alerts += 1
                alert_json = create_json_alert(
                    patient_id, window_end_time, window_data, prediction_proba
                )
                socketio.emit('new_alert', alert_json)
                
                # Tell the frontend to SHOW the snooze button
                socketio.emit('activate_alarm')
        else:
            # Patient risk has dropped. Automatically reset the snooze.
            if snooze_is_active:
                print("Patient risk dropped. Resetting snooze.")
            snooze_is_active = False
            # Tell the frontend to HIDE the snooze button
            socketio.emit('deactivate_alarm')
        # --- END OF UPGRADED SNOOZE LOGIC ---
            
        socketio.sleep(0.5) # 500ms per row

    if simulation_running:
        print(f"Simulation complete. Total alerts: {triggered_alerts}")
        socketio.emit('status_update', {'msg': f'Simulation Finished. Total alerts: {triggered_alerts}'})
        socketio.emit('deactivate_alarm') # Hide snooze button
        socketio.emit('simulation_ended')
        simulation_running = False

# --- 6. RUN THE WEB SERVER ---
if __name__ == '__main__':
    print("Starting Flask server... Open http://127.0.0.1:5000 in your browser.")
    socketio.run(app, host='0.0.0.0', port=5000)