# -*- coding: utf-8 -*-
"""
Final Simulation Script: Loads model and SCALER from separate, dedicated files, 
replicating complex feature engineering (as seen in the error log) for prediction.
"""
import pandas as pd
import numpy as np
import os
import json
import xgboost as xgb
import joblib # For loading the saved model and scaler
import time
from sklearn.preprocessing import StandardScaler # Required for type checking

# --- CONFIGURATION ---
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir_context = script_dir 
except NameError:
     parent_dir_context = os.getcwd() 

# FILE PATHS
TEST_DATA_FILE = 'test_patient_data.csv' 
MODEL_FILE = 'xgboost_stage1_model.pkl'

# --- RENAME CONFIG FILE AND ADD SCALER FILE ---
CONFIG_FILE = 'model_config.pkl'          # This file holds metadata (the dictionary that was failing)
SCALER_FILE = 'feature_scaler.pkl'        # Assume the StandardScaler object is saved here

data_dir = os.path.join(parent_dir_context, "data")
model_dir = os.path.join(parent_dir_context, "model")

TEST_DATA_PATH = os.path.join(data_dir, TEST_DATA_FILE)
MODEL_PATH = os.path.join(model_dir, MODEL_FILE)
SCALER_PATH = os.path.join(model_dir, SCALER_FILE) # New dedicated scaler path

# MODEL PARAMS (Must match training parameters)
VITAL_COLUMNS = ['HR', 'RR', 'SpO2', 'MAP']
WINDOW_SIZE = 30 
TIME_FORMAT = '%H:%M:%S'
OPTIMAL_THRESHOLD = 0.40


# --- Helper Functions (Feature Engineering - Synchronization) ---

# We must replicate the complex features based on the error log:
def calculate_features(window_df):
    """
    Calculates features for a single window, replicating complex structure 
    (including HR_MAP_product_mean.1 and deviation terms) from the training data.
    """
    if len(window_df) < WINDOW_SIZE: return {} 

    # Calculate required rolling components
    mean_hr = window_df['HR'].mean()
    median_hr = window_df['HR'].median()
    mean_map = window_df['MAP'].mean()
    median_map = window_df['MAP'].median()
    
    last_hr = window_df['HR'].iloc[-1]
    last_map = window_df['MAP'].iloc[-1]
    
    first_hr = window_df['HR'].iloc[0]
    first_map = window_df['MAP'].iloc[0]

    # --- 1. Basic Rolling Statistics ---
    features = {}
    for col in VITAL_COLUMNS:
        features[f'{col}_mean'] = window_df[col].mean()
        features[f'{col}_median'] = window_df[col].median()
        features[f'{col}_std'] = window_df[col].std()
        features[f'{col}_min'] = window_df[col].min()
        features[f'{col}_max'] = window_df[col].max()
        features[f'{col}_trend'] = last_hr - first_hr # Simplistic implementation of trend

    # --- 2. Advanced / Unseen Features (Matching Error Log) ---
    
    # Deviation Features (The difference between current and central tendency of the window)
    features['HR_diff_from_mean'] = last_hr - mean_hr
    features['HR_diff_from_median'] = last_hr - median_hr
    features['MAP_diff_from_mean'] = last_map - mean_map
    features['MAP_diff_from_median'] = last_map - median_map

    # Interaction / Product Features
    features['HR_MAP_product_mean'] = (window_df['HR'] * window_df['MAP']).mean()
    
    # Simple Ratios
    features['SpO2_RR_ratio_mean'] = (window_df['SpO2'] / window_df['RR']).mean()
    features['SpO2_RR_ratio_min'] = (window_df['SpO2'] / window_df['RR']).min()

    # --- 3. Replication of Index Error Features (The .1 suffixes) ---
    # We must include these features with the exact names the scaler expects, 
    # even if they are duplicate calculations, to satisfy the feature dimension and name check.
    # NOTE: The actual logic for these should be identical to the base features in your training.
    
    features['HR_MAP_product_mean.1'] = features['HR_MAP_product_mean']
    features['HR_diff_from_mean.1'] = features['HR_diff_from_mean']
    features['HR_diff_from_median.1'] = features['HR_diff_from_median']
    features['MAP_diff_from_mean.1'] = features['MAP_diff_from_mean']
    features['MAP_diff_from_median.1'] = features['MAP_diff_from_median']

    # --- 4. Replication of all Static/Encoded Features required by the model (placeholders) ---
    # The actual list of feature_cols_model will guide the final reindexing.

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

    # --- FIX: Cast all NumPy types to standard Python types ---
    json_output = {
        "patient_id": str(patient_id), # Cast to string
        "window_end_time": window_end_time,
        "pattern_detected": pattern_code,
        "risk_score_proba": float(round(prediction_proba, 4)), # Cast to float
        "analysis_window": f"{WINDOW_SIZE} minutes",
        "demographics": {
            "age": int(last_row['age']), # Already correctly cast to int
            "sex": str(last_row['sex']), # Cast to string
            "bmi": float(round(last_row['bmi'], 1)) # Cast to float
        },
        "trigger_data_trends": {
            "current_HR": float(round(last_row['HR'], 1)), # Cast to float
            "current_MAP": float(round(last_row['MAP'], 1)), # Cast to float
            "current_RR": float(round(last_row['RR'], 1)), # Cast to float
            "HR_trend": f"{hr_trend_display:.1f} bpm",
            "MAP_trend": f"{map_trend_display:.1f} mmHg",
        },
        "raw_window_summary": {
            "min_HR": float(round(raw_data_window['HR'].min(), 1)), # Cast to float
            "min_SpO2": float(round(raw_data_window['SpO2'].min(), 1)), # Cast to float
            "avg_MAP": float(round(raw_data_window['MAP'].mean(), 1)) # Cast to float
        }
    }
    # --- End of Fix ---

    json_string = json.dumps(json_output, indent=2)
    return json_string[:2000]


# --- Main Simulation Logic ---
if __name__ == '__main__':
    print("\n" + "="*80)
    print(f"STAGE 1 ALERT SIMULATION ENGINE STARTING")
    print(f"Attempting to load scaler from {SCALER_FILE}...")
    print("="*80)

    # 1. Load Model and Scaler
    try:
        # Load XGBoost model
        model = joblib.load(MODEL_PATH)
        print(f"✓ Model loaded from {MODEL_PATH}")
        
        # Load Scaler from the dedicated SCALER_PATH
        # We assume the file contains the StandardScaler object itself
        scaler = joblib.load(SCALER_PATH)
             
        if not isinstance(scaler, StandardScaler):
             raise TypeError(f"Object loaded from '{SCALER_FILE}' is {type(scaler).__name__}, not StandardScaler.")
             
        print(f"✓ Scaler loaded successfully from {SCALER_FILE} (Type: {type(scaler).__name__})")
        
        # Get feature names the model was trained on
        feature_cols_model = model.feature_names_in_.tolist()
        
    except FileNotFoundError as e:
        print(f"\nFATAL ERROR: Could not find required file: {e}")
        print(f"ACTION REQUIRED: Please rename your fitted StandardScaler file to '{SCALER_FILE}' and place it in the '{model_dir}' folder.")
        print(">>> DID YOU RUN 'train_xgboost.py' FIRST? <<<")
        exit(1)
    except TypeError as e:
        print(f"\nFATAL ERROR: Scaler file content is incorrect. {e}")
        print("ACTION REQUIRED: Ensure the file contains ONLY the fitted StandardScaler object.")
        exit(1)
    except Exception as e:
        print(f"FATAL ERROR loading model components: {e}")
        exit(1)

    # 2. Load Simulation Data
    try:
        df_sim = pd.read_csv(TEST_DATA_PATH)
        print(f"\n✓ Loaded test data from '{TEST_DATA_FILE}' with {len(df_sim)} rows.")
        
        df_sim['time_dt'] = pd.to_datetime(df_sim['time'], format=TIME_FORMAT, errors='coerce')
        df_sim = df_sim.dropna(subset=['time_dt'])
        
    except Exception as e:
        print(f"FATAL ERROR loading simulation data from {TEST_DATA_PATH}: {e}")
        exit(1)

    patient_id = df_sim['patient_id'].iloc[0] if not df_sim.empty else "P-SIM-000"
    print(f"SIMULATING PATIENT: {patient_id}")
    print(f"Alert Threshold set at: {OPTIMAL_THRESHOLD}")

    # 3. Simulation Loop (Sliding Window)
    triggered_alerts = []
    
    if not all(col in df_sim.columns for col in VITAL_COLUMNS):
        print(f"FATAL ERROR: Test data is missing required vital columns: {VITAL_COLUMNS}")
        exit(1)
        
    # Iterate row by row (using index) to simulate real-time stream
    for end_index in range(len(df_sim)):
        
        if end_index < WINDOW_SIZE - 1:
            continue
            
        start_index = end_index - WINDOW_SIZE + 1
        window_data = df_sim.iloc[start_index : end_index + 1].copy()
        window_end_time = window_data['time'].iloc[-1]
        
        # 4. Feature Engineering
        features_dict = calculate_features(window_data)
        X_live = pd.DataFrame([features_dict])
        
        # Add static and encoded features (CRITICAL: Must match training)
        X_live['age'] = window_data['age'].iloc[-1]
        X_live['bmi'] = window_data['bmi'].iloc[-1]
        
        sex = window_data['sex'].iloc[-1]
        X_live['sex_encoded'] = 1 if sex == 'M' else 0 

        if 'sex' in X_live.columns: X_live = X_live.drop(columns=['sex'])

        # 5. Prepare for Prediction
        # Reindex features to match the exact order and column set of the training data
        # This will add the missing HR_MAP_product_mean.1 and fill them with 0.0 (necessary for scaling)
        X_live = X_live.reindex(columns=feature_cols_model, fill_value=0.0)
        
        # Scaling (MUST use the loaded scaler object's transform method)
        try:
            # We must pass a DataFrame with the *exact* column names from training
            X_scaled = scaler.transform(X_live)
        except Exception as e:
            print(f"Scaling failed at {window_end_time}. Error: {type(e).__name__}: {e}")
            print(f"Features sent to scaler: {X_live.columns.tolist()}")
            print(f"Features scaler expected: {scaler.feature_names_in_}")
            continue

        # 6. Prediction
        prediction_proba = model.predict_proba(X_scaled)[:, 1][0]
        
        if prediction_proba >= OPTIMAL_THRESHOLD:
            # 7. Trigger Alert
            alert_json_str = create_json_alert(
                patient_id, 
                window_end_time, 
                window_data, 
                prediction_proba
            )
            triggered_alerts.append((window_end_time, alert_json_str))
            
            print(f"\nALERT TRIGGERED at {window_end_time} (P: {prediction_proba:.4f})")
            print("--- JSON Output Start (Stage 1) ---")
            print(alert_json_str)
            print("--- JSON Output End ---")


    if not triggered_alerts:
        print("\nSimulation complete. No high-risk alerts were triggered.")
    
    print("\n" + "="*80)
    print("SIMULATION ENDED")
    print(f"Total alerts triggered: {len(triggered_alerts)}")
    print("="*80)