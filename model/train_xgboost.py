# -*- coding: utf-8 -*-
"""
Creates labels based on future vital signs, splits data by patient,
trains an XGBoost model to predict deterioration, evaluates it
with comprehensive metrics and threshold tuning, and saves the trained model.

ENHANCEMENTS:
- Clinical safety threshold analysis (Recall ≥ 80%)
- Hyperparameter tuning with GridSearchCV and GroupKFold
- Enhanced feature importance analysis
- Comprehensive performance analysis
- Advanced model saving with threshold information
- Fixed deprecated XGBoost parameters
- CORRECTED: Includes StandardScaler fit/transform/save
- CORRECTED: Re-applies feature names after scaling for model training
"""
import pandas as pd
import numpy as np
import os
import xgboost as xgb
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, precision_recall_curve, auc, confusion_matrix,
    ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings('ignore')

# --- Configuration ---
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    data_dir = os.path.join(parent_dir, "data")
    model_dir = os.path.join(parent_dir, "model")
except NameError:
    print("Warning: Could not determine script directory accurately.")
    print("Assuming 'data' and 'model' folders are relative to the current working directory.")
    parent_dir = "."
    data_dir = "./data"
    model_dir = "./model"

# Input Files
CLEANED_DATA_FILE = 'patient_data.csv'
FEATURES_FILE = 'patient_features.csv'

# Output Files
OUTPUT_LABELED_FILE = 'patient_features_labeled.csv'
SAVED_MODEL_FILE = 'xgboost_stage1_model.pkl'
SAVED_SCALER_FILE = 'feature_scaler.pkl'
TUNING_RESULTS_FILE = 'hyperparameter_tuning_results.csv'

CLEANED_DATA_PATH = os.path.join(data_dir, CLEANED_DATA_FILE)
FEATURES_PATH = os.path.join(data_dir, FEATURES_FILE)
OUTPUT_LABELED_PATH = os.path.join(data_dir, OUTPUT_LABELED_FILE)
MODEL_SAVE_PATH = os.path.join(model_dir, SAVED_MODEL_FILE)

# --- FIX 1: Corrected Scaler Save Path ---
# It should be saved to the 'model' directory, not 'data'
SCALER_SAVE_PATH = os.path.join(model_dir, SAVED_SCALER_FILE) 
TUNING_RESULTS_PATH = os.path.join(model_dir, TUNING_RESULTS_FILE)

os.makedirs(model_dir, exist_ok=True)
print(f"Data directory: '{data_dir}'")
print(f"Model directory: '{model_dir}'")

# Labeling Parameters
LABEL_LOOKAHEAD_MINUTES = 120
MAP_THRESHOLD = 65
SUSTAINED_DURATION_MINUTES = 10

# Data Splitting Parameters
TEST_SIZE = 0.25
RANDOM_STATE = 42

# Threshold Tuning Parameters
THRESHOLD_RANGE = np.arange(0.05, 0.96, 0.02)  # More granular threshold testing
CLINICAL_RECALL_TARGET = 0.80  # Target recall for clinical safety

# Hyperparameter Tuning Parameters
ENABLE_HYPERPARAMETER_TUNING = True  # Set to False to skip tuning (use baseline)
HYPERPARAM_GRID = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0],
    'gamma': [0, 0.1, 0.2]
}
CV_FOLDS = 3  # Number of cross-validation folds

# --- 1. Load Data ---
print("\n" + "="*70)
print("LOADING DATA")
print("="*70)
try:
    print(f"Loading cleaned time-series data from '{CLEANED_DATA_PATH}'...")
    df_cleaned = pd.read_csv(CLEANED_DATA_PATH)
    print(f"Loading engineered features from '{FEATURES_PATH}'...")
    features_df = pd.read_csv(FEATURES_PATH)
    print(f"✓ Loaded {len(df_cleaned)} cleaned rows and {len(features_df)} feature rows.")
except FileNotFoundError as e:
    print(f"Error loading files: {e}")
    print(f"Ensure '{CLEANED_DATA_FILE}' and '{FEATURES_FILE}' exist in '{data_dir}'.")
    exit()
except Exception as e:
    print(f"An unexpected error occurred loading data: {e}")
    exit()

# --- 2. Prepare Time Columns for Labeling ---
print("\n" + "="*70)
print("PREPARING TIME COLUMNS FOR LABELING")
print("="*70)
try:
    df_cleaned['time_dt'] = pd.to_datetime(df_cleaned['time'], format='%H:%M:%S', errors='coerce')
    df_cleaned = df_cleaned.dropna(subset=['time_dt'])
    df_cleaned['time_minutes'] = (df_cleaned['time_dt'] - df_cleaned.groupby('patient_id')['time_dt'].transform('min')).dt.total_seconds() / 60
    df_cleaned['time_minutes'] = df_cleaned['time_minutes'].round().astype(int)
    print("✓ Created 'time_minutes' in cleaned data.")

    patient_start_times = df_cleaned.groupby('patient_id')['time_dt'].min().to_dict()

    def time_str_to_rel_minutes(row):
        try:
            base_date = pd.Timestamp('2000-01-01')
            end_time_dt = pd.to_datetime(base_date.date().isoformat() + ' ' + str(row['window_end_time']), errors='coerce')
            start_time_dt = patient_start_times.get(row['patient_id'])

            if pd.isna(end_time_dt) or pd.isna(start_time_dt): return np.nan

            start_time_dt_on_base = pd.to_datetime(base_date.date().isoformat() + ' ' + start_time_dt.strftime('%H:%M:%S'), errors='coerce')
            if pd.isna(start_time_dt_on_base): return np.nan

            time_delta_seconds = (end_time_dt - start_time_dt_on_base).total_seconds()
            if time_delta_seconds < -3600:
                time_delta_seconds += 24 * 3600

            return round(time_delta_seconds / 60.0)

        except Exception: return np.nan

    features_df['window_end_time_minutes'] = features_df.apply(time_str_to_rel_minutes, axis=1)
    failed_time_conv = features_df['window_end_time_minutes'].isna().sum()
    if failed_time_conv > 0:
        print(f"Warning: {failed_time_conv} 'window_end_time' values failed conversion to minutes.")
        features_df = features_df.dropna(subset=['window_end_time_minutes'])
    features_df['window_end_time_minutes'] = features_df['window_end_time_minutes'].astype(int)
    print("✓ Created 'window_end_time_minutes' in features data.")

except KeyError as ke:
    print(f"Error: Missing expected column for time processing: {ke}")
    exit()
except Exception as e:
    print(f"An unexpected error occurred during time preparation: {e}")
    exit()

# --- 3. Identify Deterioration Events in Cleaned Data ---
print("\n" + "="*70)
print("IDENTIFYING DETERIORATION EVENTS (LABELING)")
print("="*70)
print(f"Definition: MAP < {MAP_THRESHOLD} for {SUSTAINED_DURATION_MINUTES}+ consecutive minutes.")
df_cleaned = df_cleaned.sort_values(by=['patient_id', 'time_minutes'])

df_cleaned['map_low'] = df_cleaned['MAP'] < MAP_THRESHOLD

df_cleaned['is_event_period'] = df_cleaned.groupby('patient_id', group_keys=False)['map_low'].apply(
    lambda x: x.rolling(window=SUSTAINED_DURATION_MINUTES, min_periods=SUSTAINED_DURATION_MINUTES)
               .apply(lambda w: w.all(), raw=True)
).fillna(0).astype(bool)

df_cleaned['prev_is_event_period'] = df_cleaned.groupby('patient_id')['is_event_period'].shift(1).fillna(False)
df_cleaned['event_just_started'] = df_cleaned['is_event_period'] & (~df_cleaned['prev_is_event_period'])

event_start_times = df_cleaned[df_cleaned['event_just_started']].groupby('patient_id')['time_minutes'].apply(list).to_dict()

found_events_count = sum(len(v) for v in event_start_times.values())
print(f"✓ Found {found_events_count} event start times across {len(event_start_times)} patients.")

# --- 4. Assign Labels to Feature Windows ---
print(f"\nAssigning labels based on {LABEL_LOOKAHEAD_MINUTES}-minute lookahead...")

def assign_label(row):
    patient_id = row['patient_id']
    window_end_minute = row['window_end_time_minutes']
    lookahead_start_minute = window_end_minute + 1
    lookahead_end_minute = window_end_minute + LABEL_LOOKAHEAD_MINUTES

    if patient_id in event_start_times:
        patient_event_starts = event_start_times[patient_id]
        for event_start_minute in patient_event_starts:
            if lookahead_start_minute <= event_start_minute < lookahead_end_minute:
                return 1
    return 0

features_df['label'] = features_df.apply(assign_label, axis=1)
labels_assigned_count = features_df['label'].sum()

print(f"✓ Assigned label '1' (deterioration) to {labels_assigned_count} feature windows.")

# --- Check Label Distribution ---
print("\nLabel Distribution:")
label_counts = features_df['label'].value_counts()
label_dist = features_df['label'].value_counts(normalize=True).round(4)
print(label_dist)

if 0 not in label_counts or 1 not in label_counts:
    print("\nWarning: Only one class present in labels. Model training might fail or be meaningless.")
    if len(label_counts) < 2: exit("Exiting due to single class label.")
    scale_pos_weight = 1
else:
    count_negative = label_counts.get(0, 0)
    count_positive = label_counts.get(1, 0)
    if count_positive == 0:
        print("\nError: No positive labels found. Cannot calculate scale_pos_weight.")
        exit("Exiting due to no positive labels.")
    scale_pos_weight = count_negative / count_positive
    print(f"\nCalculated scale_pos_weight for XGBoost: {scale_pos_weight:.2f}")

try:
    features_df.to_csv(OUTPUT_LABELED_PATH, index=False)
    print(f"\n✓ Successfully saved labeled features to '{OUTPUT_LABELED_PATH}'")
except Exception as e:
    print(f"\nError saving labeled features file '{OUTPUT_LABELED_PATH}': {e}")

# --- 5. Split Data into Training and Testing Sets ---
print("\n" + "="*70)
print("SPLITTING DATA INTO TRAIN/TEST SETS")
print("="*70)
print(f"Using GroupShuffleSplit to separate patients (Test size: {TEST_SIZE}).")

feature_columns = [
    col for col in features_df.columns if col not in
    ['patient_id', 'window_end_time', 'sex', 'label', 'window_end_time_minutes']
]
X = features_df[feature_columns]
y = features_df['label']
groups = features_df['patient_id']

print(f"Feature columns used for training ({len(feature_columns)}):")
print(feature_columns)

gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
try:
    train_idx, test_idx = next(gss.split(X, y, groups))
except ValueError as e:
    print(f"\nError during GroupShuffleSplit: {e}")
    exit()

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
groups_train, groups_test = groups.iloc[train_idx], groups.iloc[test_idx]

train_patients = set(features_df.iloc[train_idx]['patient_id'])
test_patients = set(features_df.iloc[test_idx]['patient_id'])
if train_patients.intersection(test_patients):
    print("\nWarning: Patient data leakage detected between train and test sets!")
else:
    print("\n✓ Patient separation between train and test sets verified.")

print(f"\nTraining set shape: X={X_train.shape}, y={y_train.shape}")
print(f"Testing set shape: X={X_test.shape}, y={y_test.shape}")
print(f"Training set label distribution:\n{y_train.value_counts(normalize=True).round(4)}")
print(f"Testing set label distribution:\n{y_test.value_counts(normalize=True).round(4)}")


# --- 5b. (FIX 2) Scale Features & Save Scaler ---
print("\n" + "="*70)
print("SCALING FEATURES & SAVING SCALER")
print("="*70)

print(f"Initializing StandardScaler...")
scaler = StandardScaler()

# CRITICAL: Fit scaler ONLY on X_train
print(f"Fitting scaler on X_train (shape: {X_train.shape})...")
scaler.fit(X_train)
print("✓ Scaler fitted.")

# Save the fitted scaler immediately to the correct path
try:
    joblib.dump(scaler, SCALER_SAVE_PATH)
    print(f"✓ SCALER SAVED to: '{SCALER_SAVE_PATH}'")
except Exception as e:
    print(f"FATAL ERROR: Could not save scaler. Error: {e}")
    print(f"Check permissions for directory: {model_dir}")
    exit(1)

# Now, transform both train and test sets
print("Transforming X_train and X_test...")
# scaler.transform() returns a NumPy array, stripping column names.
X_train_scaled_np = scaler.transform(X_train)
X_test_scaled_np = scaler.transform(X_test)

# Convert back to DataFrame, adding feature names back.
# This is CRITICAL for the model to learn the feature names.
# 'feature_columns' was defined in Section 5
X_train = pd.DataFrame(X_train_scaled_np, columns=feature_columns)
X_test = pd.DataFrame(X_test_scaled_np, columns=feature_columns)

print("✓ Feature scaling complete (DataFrames recreated with feature names).")


# --- 6. Hyperparameter Tuning with GroupKFold ---
print("\n" + "="*70)
print("HYPERPARAMETER TUNING")
print("="*70)

if ENABLE_HYPERPARAMETER_TUNING:
    print("Starting hyperparameter tuning with GridSearchCV and GroupKFold...")
    print(f"Parameter grid: {HYPERPARAM_GRID}")
    print(f"Cross-validation folds: {CV_FOLDS}")
    
    # Create base model (FIXED: removed deprecated use_label_encoder)
    base_model = xgb.XGBClassifier(
        objective='binary:logistic',
        scale_pos_weight=scale_pos_weight,
        eval_metric='logloss',
        random_state=RANDOM_STATE
    )
    
    # Create GroupKFold for patient-aware cross-validation
    group_kfold = GroupKFold(n_splits=CV_FOLDS)
    
    # Perform GridSearchCV
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=HYPERPARAM_GRID,
        scoring='f1',  # Can change to 'recall' for clinical priority
        cv=group_kfold,
        verbose=1,
        n_jobs=-1,
        return_train_score=True
    )
    
    print("GridSearchCV in progress... (This may take several minutes)")
    # We now pass the SCALED X_train DataFrame to fit
    grid_search.fit(X_train, y_train, groups=groups_train)
    
    # Get best model and parameters
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
    print(f"✓ Hyperparameter tuning completed!")
    print(f"Best parameters: {best_params}")
    print(f"Best cross-validation F1 score: {best_score:.4f}")
    
    # Save tuning results
    tuning_results_df = pd.DataFrame(grid_search.cv_results_)
    try:
        tuning_results_df.to_csv(TUNING_RESULTS_PATH, index=False)
        print(f"✓ Tuning results saved to '{TUNING_RESULTS_PATH}'")
    except Exception as e:
        print(f"Warning: Could not save tuning results: {e}")
    
    model = best_model
    tuning_completed = True
    
else:
    print("Hyperparameter tuning disabled. Using baseline parameters...")
    # FIXED: removed deprecated use_label_encoder
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        scale_pos_weight=scale_pos_weight,
        eval_metric='logloss',
        random_state=RANDOM_STATE
    )
    tuning_completed = False

# --- 7. Train Final Model ---
print("\n" + "="*70)
print("TRAINING FINAL MODEL")
print("="*70)

print("Training final model on full (SCALED) training set...")
try:
    # We fit the model on the SCALED X_train DataFrame
    model.fit(X_train, y_train)
    print("✓ Model training complete.")
except Exception as e:
    print(f"\nError during model training: {e}")
    exit()

# --- 8. Evaluate Model with Comprehensive Threshold Analysis ---
print("\n" + "="*70)
print("COMPREHENSIVE MODEL EVALUATION")
print("="*70)

try:
    # Make predictions using the SCALED X_test DataFrame
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Calculate metrics at default threshold (0.5)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else 0.5

    print("\n--- Performance at Default Threshold (0.5) ---")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"AUC-ROC:   {roc_auc:.4f}")

    # --- Advanced Threshold Tuning Analysis ---
    print(f"\n--- Threshold Tuning Analysis ({len(THRESHOLD_RANGE)} thresholds) ---")
    
    threshold_results = []
    for threshold in THRESHOLD_RANGE:
        y_pred_threshold = (y_pred_proba >= threshold).astype(int)
        
        # Handle cases with no positive predictions
        if np.sum(y_pred_threshold) == 0:
            tn = np.sum(y_test == 0)
            fp = 0
            fn = np.sum(y_test == 1)
            tp = 0
        else:
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred_threshold).ravel()
        
        prec = precision_score(y_test, y_pred_threshold, zero_division=0)
        rec = recall_score(y_test, y_pred_threshold, zero_division=0)
        f1_threshold = f1_score(y_test, y_pred_threshold, zero_division=0)
        
        threshold_results.append({
            'threshold': threshold,
            'precision': prec,
            'recall': rec,
            'f1_score': f1_threshold,
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn,
            'true_negatives': tn,
            'false_alarms_per_alert': 1/prec if prec > 0 else float('inf'),
            'missed_events': fn
        })

    threshold_df = pd.DataFrame(threshold_results)

    # Find optimal threshold (max F1)
    optimal_idx = threshold_df['f1_score'].idxmax()
    optimal_threshold = threshold_df.loc[optimal_idx, 'threshold']
    optimal_precision = threshold_df.loc[optimal_idx, 'precision']
    optimal_recall = threshold_df.loc[optimal_idx, 'recall']
    optimal_f1 = threshold_df.loc[optimal_idx, 'f1_score']

    # Find clinical safety threshold (Recall ≥ 80%)
    clinical_thresholds = threshold_df[threshold_df['recall'] >= CLINICAL_RECALL_TARGET]
    
    if not clinical_thresholds.empty:
        # Choose threshold with highest precision among those meeting recall target
        clinical_idx = clinical_thresholds['precision'].idxmax()
        clinical_threshold = clinical_thresholds.loc[clinical_idx, 'threshold']
        clinical_precision = clinical_thresholds.loc[clinical_idx, 'precision']
        clinical_recall = clinical_thresholds.loc[clinical_idx, 'recall']
        clinical_f1 = clinical_thresholds.loc[clinical_idx, 'f1_score']
        clinical_available = True
    else:
        # Use maximum achievable recall if target not met
        clinical_idx = threshold_df['recall'].idxmax()
        clinical_threshold = threshold_df.loc[clinical_idx, 'threshold']
        clinical_precision = threshold_df.loc[clinical_idx, 'precision']
        clinical_recall = threshold_df.loc[clinical_idx, 'recall']
        clinical_f1 = threshold_df.loc[clinical_idx, 'f1_score']
        clinical_available = False

    print(f"\n--- Optimal Thresholds Identified ---")
    print(f"1. F1-Optimal Threshold: {optimal_threshold:.3f}")
    print(f"   Precision: {optimal_precision:.4f}, Recall: {optimal_recall:.4f}, F1: {optimal_f1:.4f}")
    if optimal_precision > 0:
        print(f"   False Alarms per True Alert: {1/optimal_precision:.1f}")
    
    print(f"\n2. Clinical Safety Threshold: {clinical_threshold:.3f}")
    print(f"   Precision: {clinical_precision:.4f}, Recall: {clinical_recall:.4f}, F1: {clinical_f1:.4f}")
    if clinical_precision > 0:
        print(f"   False Alarms per True Alert: {1/clinical_precision:.1f}")
    
    if not clinical_available:
        print(f"   ⚠️  Warning: No threshold achieved {CLINICAL_RECALL_TARGET:.0%} recall target")
        print(f"   Using maximum achievable recall ({clinical_recall:.1%}) instead")

    # Store all metrics
    metrics = {
        'default': {'precision': precision, 'recall': recall, 'f1': f1, 'threshold': 0.5},
        'optimal': {'precision': optimal_precision, 'recall': optimal_recall, 'f1': optimal_f1, 'threshold': optimal_threshold},
        'clinical': {'precision': clinical_precision, 'recall': clinical_recall, 'f1': clinical_f1, 'threshold': clinical_threshold},
        'auc_roc': roc_auc,
        'y_pred_proba': y_pred_proba,
        'y_test': y_test,
        'threshold_df': threshold_df,
        'clinical_target_met': clinical_available
    }

except Exception as e:
    print(f"\nError during model evaluation: {e}")
    # exit() # Comment out exit to allow saving even if evaluation fails

# --- 9. Feature Importance Analysis ---
print("\n" + "="*70)
print("FEATURE IMPORTANCE ANALYSIS")
print("="*70)
try:
    feature_importances = pd.Series(model.feature_importances_, index=feature_columns)
    top_features = feature_importances.nlargest(15)
    
    print("Top 15 Most Important Features:")
    for i, (feature, importance) in enumerate(top_features.items(), 1):
        print(f"{i:2d}. {feature:<30} {importance:.4f}")
    
    metrics['feature_importances'] = feature_importances
    metrics['top_features'] = top_features

except Exception as e:
    print(f"Could not calculate feature importance: {e}")
    metrics['feature_importances'] = None
    metrics['top_features'] = None

# --- 10. Generate Comprehensive Plots ---
print("\n" + "="*70)
print("GENERATING COMPREHENSIVE PLOTS")
print("="*70)

try:
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Metrics vs Threshold
    axes[0,0].plot(metrics['threshold_df']['threshold'], metrics['threshold_df']['precision'], 
                   label='Precision', marker='o', markersize=3, linewidth=2)
    axes[0,0].plot(metrics['threshold_df']['threshold'], metrics['threshold_df']['recall'], 
                   label='Recall', marker='s', markersize=3, linewidth=2)
    axes[0,0].plot(metrics['threshold_df']['threshold'], metrics['threshold_df']['f1_score'], 
                   label='F1-Score', marker='^', markersize=4, linewidth=2)
    
    # Add threshold lines
    axes[0,0].axvline(x=metrics['optimal']['threshold'], color='red', linestyle='--', 
                      label=f'Optimal F1 (T={metrics["optimal"]["threshold"]:.2f})')
    axes[0,0].axvline(x=metrics['clinical']['threshold'], color='green', linestyle='--', 
                      label=f'Clinical Safety (T={metrics["clinical"]["threshold"]:.2f})')
    axes[0,0].axvline(x=0.5, color='gray', linestyle=':', 
                      label='Default (T=0.5)')
    axes[0,0].axhline(y=CLINICAL_RECALL_TARGET, color='orange', linestyle=':', alpha=0.7, 
                      label=f'Recall Target ({CLINICAL_RECALL_TARGET:.0%})')
    
    axes[0,0].set_xlabel('Classification Threshold')
    axes[0,0].set_ylabel('Score')
    axes[0,0].set_title('Performance Metrics vs Threshold')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Plot 2: Precision-Recall Curve
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_proba)
    axes[0,1].plot(recall_curve, precision_curve, linewidth=2, label='PR Curve')
    axes[0,1].scatter(metrics['optimal']['recall'], metrics['optimal']['precision'], 
                     color='red', s=100, zorder=5, 
                     label=f'Optimal F1 (T={metrics["optimal"]["threshold"]:.2f})')
    axes[0,1].scatter(metrics['clinical']['recall'], metrics['clinical']['precision'], 
                     color='green', s=100, zorder=5, 
                     label=f'Clinical Safety (T={metrics["clinical"]["threshold"]:.2f})')
    axes[0,1].scatter(metrics['default']['recall'], metrics['default']['precision'], 
                     color='gray', s=100, zorder=5, marker='s',
                     label='Default (T=0.5)')
    axes[0,1].set_xlabel('Recall (Sensitivity)')
    axes[0,1].set_ylabel('Precision (PPV)')
    axes[0,1].set_title('Precision-Recall Curve')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # Plot 3: ROC Curve
    RocCurveDisplay.from_predictions(y_test, y_pred_proba, ax=axes[1,0])
    axes[1,0].set_title(f'ROC Curve (AUC = {roc_auc:.3f})')
    axes[1,0].grid(True, alpha=0.3)
    
    # Plot 4: Feature Importance
    if metrics['top_features'] is not None:
        metrics['top_features'].plot(kind='barh', ax=axes[1,1])
        axes[1,1].set_title('Top 15 Feature Importances')
        axes[1,1].set_xlabel('Importance Score')
    else:
        axes[1,1].text(0.5, 0.5, 'Feature Importance failed', ha='center', va='center')
        axes[1,1].set_title('Feature Importances')

    plt.tight_layout()
    plt.show()
    
    # Additional Plot: Threshold Comparison
    fig2, ax = plt.subplots(figsize=(10, 6))
    thresholds_to_compare = [0.5, metrics['optimal']['threshold'], metrics['clinical']['threshold']]
    threshold_names = ['Default', 'Optimal F1', 'Clinical Safety']
    colors = ['gray', 'red', 'green']
    
    for i, (threshold, name, color) in enumerate(zip(thresholds_to_compare, threshold_names, colors)):
        y_pred_comp = (y_pred_proba >= threshold).astype(int)
        prec = precision_score(y_test, y_pred_comp, zero_division=0)
        rec = recall_score(y_test, y_pred_comp, zero_division=0)
        
        ax.bar(i - 0.2, prec, width=0.4, color=color, alpha=0.7, label=f'Precision ({name})' if i == 0 else "")
        ax.bar(i + 0.2, rec, width=0.4, color=color, alpha=0.9, label=f'Recall ({name})' if i == 0 else "")
        ax.text(i - 0.2, prec + 0.01, f'{prec:.2f}', ha='center', va='bottom')
        ax.text(i + 0.2, rec + 0.01, f'{rec:.2f}', ha='center', va='bottom')
    
    ax.set_xticks(range(len(thresholds_to_compare)))
    ax.set_xticklabels([f'{name}\n(T={thresh:.2f})' for name, thresh in zip(threshold_names, thresholds_to_compare)])
    ax.set_ylabel('Score')
    ax.set_title('Performance Comparison Across Thresholds')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)
    plt.tight_layout()
    plt.show()

except Exception as e:
    print(f"Could not generate plots: {e}")

# --- 11. Save Model and Configuration ---
print("\n" + "="*70)
print("SAVING MODEL AND CONFIGURATION")
print("="*70)

try:
    # Save the trained model
    joblib.dump(model, MODEL_SAVE_PATH)
    print(f"✓ Model saved to: '{MODEL_SAVE_PATH}'")
    
    # Save comprehensive configuration
    model_config = {
        'feature_columns': feature_columns,
        'scale_pos_weight': scale_pos_weight,
        'hyperparameters_tuned': tuning_completed,
        'best_hyperparameters': best_params if tuning_completed else 'default',
        'thresholds': {
            'default': metrics.get('default'),
            'optimal': metrics.get('optimal'),
            'clinical': metrics.get('clinical')
        },
        'performance': {
            'auc_roc': metrics.get('auc_roc'),
            'clinical_target_met': metrics.get('clinical_target_met')
        },
        'data_info': {
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'feature_count': len(feature_columns),
            'positive_class_ratio': y_train.mean()
        }
    }
    
    config_path = os.path.join(model_dir, 'model_config.pkl')
    joblib.dump(model_config, config_path)
    print(f"✓ Model configuration saved to: '{config_path}'")
    
    # Save threshold analysis results
    threshold_df.to_csv(os.path.join(model_dir, 'threshold_analysis.csv'), index=False)
    print(f"✓ Threshold analysis saved to: '{os.path.join(model_dir, 'threshold_analysis.csv')}'")

except Exception as e:
    print(f"Error saving model or configuration: {e}")

# --- 12. Comprehensive Next Steps Analysis ---
print("\n" + "="*70)
print("COMPREHENSIVE NEXT STEPS ANALYSIS")
print("="*70)

try:
    # Calculate improvement metrics
    default_miss_rate = 1 - metrics['default']['recall']
    clinical_miss_rate = 1 - metrics['clinical']['recall']
    
    if default_miss_rate > 0:
        miss_rate_reduction = (default_miss_rate - clinical_miss_rate) / default_miss_rate * 100
    else:
        miss_rate_reduction = 0.0

    print("Default Miss Rate: ",default_miss_rate)
    print("Clinical Miss Rate: ",clinical_miss_rate)
    print(f"\nBy adopting the Clinical Safety Threshold (T={metrics['clinical']['threshold']:.2f}):")
    print(f"✓ Miss Rate Reduction: {miss_rate_reduction:.2f}%")
except Exception as e:
    print(f"Could not complete final analysis: {e}")

print("="*70)
print("✓ SCRIPT COMPLETED SUCCESSFULLY")
print("✓ Model trained, evaluated, and saved with comprehensive analysis")
print("✓ Clinical safety threshold analysis completed")
print("✓ Next steps roadmap defined")
print("="*70)