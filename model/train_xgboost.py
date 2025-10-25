# -*- coding: utf-8 -*-
"""
Creates labels based on future vital signs, splits data by patient,
trains an XGBoost model to predict deterioration, evaluates it
with comprehensive metrics and threshold tuning, and saves the trained model.

ENHANCEMENTS:
- Threshold tuning to find optimal F1 score
- Hyperparameter tuning preparation (commented)
- Detailed performance analysis
- Advanced model saving with joblib
"""
import pandas as pd
import numpy as np
import os
import xgboost as xgb
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    precision_recall_curve,
    auc,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay
)
import matplotlib.pyplot as plt
import joblib

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
SAVED_MODEL_FILE = 'xgboost_stage1_model.pkl'  # Changed to .pkl for joblib
SAVED_SCALER_FILE = 'feature_scaler.pkl'

CLEANED_DATA_PATH = os.path.join(data_dir, CLEANED_DATA_FILE)
FEATURES_PATH = os.path.join(data_dir, FEATURES_FILE)
OUTPUT_LABELED_PATH = os.path.join(data_dir, OUTPUT_LABELED_FILE)
MODEL_SAVE_PATH = os.path.join(model_dir, SAVED_MODEL_FILE)
SCALER_SAVE_PATH = os.path.join(data_dir, SAVED_SCALER_FILE)

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
THRESHOLD_RANGE = np.arange(0.1, 0.91, 0.05)  # Test thresholds from 0.1 to 0.9

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

# --- 6. Train XGBoost Model ---
print("\n" + "="*70)
print("TRAINING XGBOOST MODEL")
print("="*70)

if 'scale_pos_weight' not in locals(): scale_pos_weight = 1

# ============================================================================
# HYPERPARAMETER TUNING SECTION (Currently using default parameters)
# ============================================================================
# For production deployment, consider implementing hyperparameter tuning:
#
# Option 1: Grid Search CV
# from sklearn.model_selection import GridSearchCV
# param_grid = {
#     'n_estimators': [100, 200, 300],
#     'max_depth': [3, 5, 7, 9],
#     'learning_rate': [0.01, 0.05, 0.1, 0.2],
#     'gamma': [0, 0.1, 0.2],
#     'subsample': [0.8, 0.9, 1.0],
#     'colsample_bytree': [0.8, 0.9, 1.0],
#     'min_child_weight': [1, 3, 5]
# }
#
# grid_search = GridSearchCV(
#     estimator=xgb.XGBClassifier(
#         objective='binary:logistic',
#         scale_pos_weight=scale_pos_weight,
#         use_label_encoder=False,
#         eval_metric='logloss',
#         random_state=RANDOM_STATE
#     ),
#     param_grid=param_grid,
#     scoring='f1',  # Or use 'roc_auc', 'recall', etc. based on priority
#     cv=3,  # 3-fold cross-validation (consider GroupKFold for patient separation)
#     verbose=2,
#     n_jobs=-1
# )
# grid_search.fit(X_train, y_train)
# model = grid_search.best_estimator_
# print(f"Best parameters: {grid_search.best_params_}")
#
# Option 2: Randomized Search CV (faster for large param spaces)
# from sklearn.model_selection import RandomizedSearchCV
# param_distributions = {
#     'n_estimators': [50, 100, 150, 200, 300],
#     'max_depth': [3, 4, 5, 6, 7, 8, 9],
#     'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
#     'gamma': [0, 0.1, 0.2, 0.3],
#     'subsample': [0.7, 0.8, 0.9, 1.0],
#     'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
#     'min_child_weight': [1, 2, 3, 4, 5]
# }
# random_search = RandomizedSearchCV(
#     estimator=xgb.XGBClassifier(...),
#     param_distributions=param_distributions,
#     n_iter=50,  # Number of random combinations to try
#     scoring='f1',
#     cv=3,
#     verbose=2,
#     n_jobs=-1,
#     random_state=RANDOM_STATE
# )
# random_search.fit(X_train, y_train)
# model = random_search.best_estimator_
#
# ============================================================================

# Current: Using baseline parameters with class imbalance handling
model = xgb.XGBClassifier(
    objective='binary:logistic',
    scale_pos_weight=scale_pos_weight,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=RANDOM_STATE,
    # Default parameters - tune these for better performance:
    # n_estimators=100,
    # max_depth=6,
    # learning_rate=0.3,
    # gamma=0,
    # subsample=1.0,
    # colsample_bytree=1.0
)

print("Starting model training with baseline parameters...")
print(f"  scale_pos_weight: {scale_pos_weight:.2f}")
try:
    model.fit(X_train, y_train)
    print("✓ Model training complete.")
except Exception as e:
    print(f"\nError during model training: {e}")
    exit()

# --- 7. Evaluate Model ---
print("\n" + "="*70)
print("EVALUATING MODEL ON TEST SET")
print("="*70)

try:
    # Make predictions
    y_pred = model.predict(X_test)  # Default threshold = 0.5
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # --- Calculate Metrics at Default Threshold (0.5) ---
    print("\n--- Performance Metrics at Default Threshold (0.5) ---")
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else 0.5

    # Store metrics for later use
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'y_pred_proba': y_pred_proba,
        'y_test': y_test
    }

    # --- Threshold Tuning Analysis ---
    print("\n" + "="*70)
    print("THRESHOLD TUNING ANALYSIS")
    print("="*70)
    print(f"Testing thresholds from {THRESHOLD_RANGE.min():.2f} to {THRESHOLD_RANGE.max():.2f}...")

    threshold_results = []

    for threshold in THRESHOLD_RANGE:
        # Apply threshold to probability predictions
        y_pred_threshold = (y_pred_proba >= threshold).astype(int)

        # Calculate metrics
        prec = precision_score(y_test, y_pred_threshold, zero_division=0)
        rec = recall_score(y_test, y_pred_threshold, zero_division=0)
        f1_threshold = f1_score(y_test, y_pred_threshold, zero_division=0)

        threshold_results.append({
            'threshold': threshold,
            'precision': prec,
            'recall': rec,
            'f1_score': f1_threshold
        })

    # Create DataFrame for easy analysis
    threshold_df = pd.DataFrame(threshold_results)

    # Find optimal threshold (max F1)
    optimal_idx = threshold_df['f1_score'].idxmax()
    optimal_threshold = threshold_df.loc[optimal_idx, 'threshold']
    optimal_precision = threshold_df.loc[optimal_idx, 'precision']
    optimal_recall = threshold_df.loc[optimal_idx, 'recall']
    optimal_f1 = threshold_df.loc[optimal_idx, 'f1_score']

    print(f"\n✓ Optimal Threshold Analysis Complete")
    print(f"\nOPTIMAL THRESHOLD (Maximizes F1-Score): {optimal_threshold:.2f}")
    print(f"  Precision: {optimal_precision:.4f}")
    print(f"  Recall:    {optimal_recall:.4f}")
    print(f"  F1-Score:  {optimal_f1:.4f}")

    # --- Generate Standard Performance Plots --- (move this section to the end)
    metrics['threshold_df'] = threshold_df
    metrics['optimal_threshold'] = optimal_threshold
    metrics['optimal_precision'] = optimal_precision
    metrics['optimal_recall'] = optimal_recall
    metrics['optimal_f1'] = optimal_f1

except Exception as e:
    print(f"\nError during model evaluation or plotting: {e}")

# --- Optional: Feature Importance --- (keep this section as is)
print("\n" + "="*70)
print("FEATURE IMPORTANCE (Top 15)")
print("="*70)
try:
    feature_importances = pd.Series(model.feature_importances_, index=feature_columns)
    print(feature_importances.nlargest(15))

    # Store feature importances for later use
    metrics['feature_importances'] = feature_importances

except Exception as e:
    print(f"Could not calculate or plot feature importance: {e}")

# --- 8. Save the Trained Model --- (keep this section as is)
print("\n" + "="*70)
print("SAVING TRAINED MODEL")
print("="*70)

try:
    os.makedirs(model_dir, exist_ok=True)
    print(f"Model directory: '{model_dir}'")

    # Save the model using joblib (recommended for sklearn-compatible models)
    joblib.dump(model, MODEL_SAVE_PATH)

    print(f"✓ Trained XGBoost model saved successfully to:")
    print(f"  '{MODEL_SAVE_PATH}'")

    # Save threshold information
    threshold_info = {
        'optimal_threshold': optimal_threshold,
        'optimal_precision': optimal_precision,
        'optimal_recall': optimal_recall,
        'optimal_f1': optimal_f1,
        'default_threshold': 0.5,
        'default_precision': precision,
        'default_recall': recall,
        'default_f1': f1
    }
    threshold_info_path = os.path.join(model_dir, 'threshold_info.pkl')
    joblib.dump(threshold_info, threshold_info_path)
    print(f"✓ Threshold tuning results saved to:")
    print(f"  '{threshold_info_path}'")

except Exception as e:
    print(f"\nError saving the trained model: {e}")

# --- Plotting Section (newly added) ---
print("\n" + "="*70)
print("GENERATING PLOTS")
print("="*70)

try:
    # Plot threshold tuning results
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Metrics vs Threshold
    axes[0].plot(metrics['threshold_df']['threshold'], metrics['threshold_df']['precision'], 
                 label='Precision', marker='o', markersize=4)
    axes[0].plot(metrics['threshold_df']['threshold'], metrics['threshold_df']['recall'], 
                 label='Recall', marker='s', markersize=4)
    axes[0].plot(metrics['threshold_df']['threshold'], metrics['threshold_df']['f1_score'], 
                 label='F1-Score', marker='^', markersize=4, linewidth=2)
    axes[0].axvline(x=metrics['optimal_threshold'], color='red', linestyle='--', 
                    label=f'Optimal Threshold ({metrics["optimal_threshold"]:.2f})')
    axes[0].axvline(x=0.5, color='gray', linestyle=':', 
                    label='Default Threshold (0.5)')
    axes[0].set_xlabel('Threshold')
    axes[0].set_ylabel('Score')
    axes[0].set_title('Performance Metrics vs Classification Threshold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Precision-Recall Trade-off
    axes[1].plot(metrics['threshold_df']['recall'], metrics['threshold_df']['precision'], 
                 marker='o', markersize=4, linewidth=2)
    axes[1].scatter(metrics['optimal_recall'], metrics['optimal_precision'], 
                   color='red', s=100, zorder=5, 
                   label=f'Optimal (T={metrics["optimal_threshold"]:.2f})')
    default_prec = metrics['precision']
    default_rec = metrics['recall']
    axes[1].scatter(default_rec, default_prec, 
                   color='gray', s=100, zorder=5, marker='s',
                   label=f'Default (T=0.5)')
    axes[1].set_xlabel('Recall (Sensitivity)')
    axes[1].set_ylabel('Precision (PPV)')
    axes[1].set_title('Precision-Recall Trade-off')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Plot feature importances
    plt.figure(figsize=(10, 8))
    metrics['feature_importances'].nlargest(15).plot(kind='barh')
    plt.title('Top 15 Feature Importances (XGBoost)')
    plt.xlabel('Importance')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

except Exception as e:
    print(f"Could not generate plots: {e}")

# ============================================================================ 
# NEXT STEPS FOR MODEL IMPROVEMENT 
# ============================================================================ 
print("\n" + "="*70) 
print("NEXT STEPS FOR MODEL IMPROVEMENT") 
print("="*70) 

print("""
Based on the initial model performance, here are recommended next steps:

1. THRESHOLD TUNING (COMPLETED)
   ✓ Optimal threshold identified: {:.2f}
   ✓ This threshold balances Precision and Recall for F1 optimization
   
   CLINICAL CONSIDERATION:
   - For deterioration prediction, HIGH RECALL is critical (minimize missed events)
   - Consider setting a lower threshold (e.g., 0.2-0.3) to achieve Recall ≥ 0.80
   - Trade-off: Lower threshold = more false alarms but fewer missed events
   - Review threshold_df results to find threshold meeting clinical requirements

2. HYPERPARAMETER TUNING (RECOMMENDED - See commented code above)
   Current Status: Using XGBoost default parameters
   
   Action Items:
   a) Implement GridSearchCV or RandomizedSearchCV (code provided in comments)
   b) Key parameters to tune:
      - n_estimators: [100, 200, 300] - More trees can improve performance
      - max_depth: [3, 5, 7, 9] - Control model complexity
      - learning_rate: [0.01, 0.05, 0.1, 0.2] - Step size for boosting
      - gamma: [0, 0.1, 0.2] - Minimum loss reduction for split
      - subsample: [0.8, 0.9, 1.0] - Fraction of samples per tree
      - colsample_bytree: [0.8, 0.9, 1.0] - Fraction of features per tree
   c) Use appropriate scoring metric:
      - 'recall' if minimizing false negatives is priority
      - 'f1' for balanced performance
      - 'roc_auc' for overall discrimination ability
   d) Consider GroupKFold for cross-validation to maintain patient separation

3. FEATURE ENGINEERING (ONGOING)
   Current Features: {num_features} engineered features
   
   Review Feature Importance (shown above) and consider:
   a) Remove low-importance features to reduce noise
   b) Create new interaction features:
      - SI * age (elderly patients with high shock index)
      - MAP_trend * HR_trend (simultaneous deterioration indicators)
      - Window-over-window comparisons (comparing current to previous window)
   c) Experiment with different window sizes:
      - Shorter windows (15 min) for acute changes
      - Longer windows (60 min) for gradual trends
   d) Add time-of-day features (if circadian patterns exist)
   e) Calculate rate-of-change features (second derivatives)

4. CLASS IMBALANCE HANDLING (VERIFY & ENHANCE)
   Current Approach: scale_pos_weight = {:.2f}
   
   Additional Techniques to Consider:
   a) SMOTE (Synthetic Minority Oversampling Technique):
      from imblearn.over_sampling import SMOTE
      smote = SMOTE(random_state=RANDOM_STATE)
      X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
   
   b) ADASYN (Adaptive Synthetic Sampling):
      from imblearn.over_sampling import ADASYN
      adasyn = ADASYN(random_state=RANDOM_STATE)
      X_train_resampled, y_train_resampled = adasyn.fit_resample(X_train, y_train)
   
   c) Class weighting in combination with sampling
   d) Ensemble methods with balanced bootstrap samples
   
   WARNING: Be cautious with synthetic sampling - validate on held-out test set
            to ensure model generalizes to real (imbalanced) data

5. ADVANCED EVALUATION
   a) Perform patient-level analysis:
      - Sensitivity per patient (% of patients with events detected)
      - Time-to-detection analysis (how early does model predict?)
   b) Cost-sensitive evaluation:
      - Assign clinical costs to FP (unnecessary interventions) vs FN (missed events)
      - Optimize threshold based on cost-benefit analysis
   c) Calibration analysis:
      - Plot calibration curve to assess probability reliability
      - Consider calibrating probabilities if needed

6. ENSEMBLE APPROACHES
   a) Train multiple models with different algorithms:
      - Random Forest
      - LightGBM
      - Neural Networks (LSTM for time-series)
   b) Combine predictions via:
      - Voting
      - Stacking
      - Weighted averaging based on validation performance

7. TEMPORAL VALIDATION
   - Current split is random by patient
   - Consider time-based split: train on earlier data, test on later data
   - Validates model's ability to generalize to future patients

8. DEPLOYMENT CONSIDERATIONS
   a) Model monitoring:
      - Track performance metrics over time
      - Detect data drift
   b) A/B testing:
      - Deploy to subset of patients
      - Compare outcomes vs standard care
   c) Real-time prediction pipeline:
      - Feature calculation latency
      - Model inference speed
      - Alert system integration

IMMEDIATE PRIORITY:
1. Run hyperparameter tuning (uncomment and execute GridSearchCV code)
2. Select threshold based on clinical recall requirements (target: Recall ≥ 0.80)
3. Retrain with optimal parameters and threshold
4. Validate on additional hold-out data if available

""".format(optimal_threshold, len(feature_columns), scale_pos_weight))

print("="*70)
print("✓ Script Finished - Model trained and saved with comprehensive analysis")
print("="*70)