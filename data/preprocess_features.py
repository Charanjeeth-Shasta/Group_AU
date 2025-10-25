import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder

# ============ CONFIGURATION ============
WINDOW_SIZE = 30                        # Window size in minutes (rows)
STEP_SIZE = 1                           # Step size in minutes (rows)
INPUT_FILENAME = "patient_data.csv"
OUTPUT_FILENAME = "patient_features.csv"
APPLY_SCALING = True                    # Apply StandardScaler to numeric features
ENCODE_CATEGORICAL = True               # Encode categorical variables (sex)
# =======================================

# Setup paths
project_root = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(project_root)
input_path = os.path.join(data_dir, INPUT_FILENAME)
output_path = os.path.join(data_dir, OUTPUT_FILENAME)

print("="*70)
print("VITAL SIGNS FEATURE ENGINEERING - SLIDING WINDOW APPROACH")
print("="*70)

# ============================================================================
# STEP 1: LOAD CLEANED DATA
# ============================================================================
print("\n1. Loading cleaned data...")
try:
    df = pd.read_csv(input_path)
    print(f"   ✓ Loaded {len(df)} rows from {INPUT_FILENAME}")
    print(f"   ✓ Found {df['patient_id'].nunique()} unique patients")
except FileNotFoundError:
    print(f"   ✗ Error: {INPUT_FILENAME} not found in {data_dir}")
    print(f"   Please run data_fetcher.py first to generate the cleaned data.")
    exit(1)

# Verify required columns exist
required_cols = ['patient_id', 'time', 'age', 'sex', 'bmi', 'HR', 'RR', 'SpO2', 'MAP']
missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    print(f"   ✗ Error: Missing required columns: {missing_cols}")
    exit(1)

print(f"   ✓ All required columns present")

# ============================================================================
# STEP 2: CONVERT TIME COLUMN
# ============================================================================
print("\n2. Converting time column...")
# The time column is in format "HH:MM:SS", convert to minutes from start for each patient
def time_to_minutes(time_str):
    """Convert HH:MM:SS to total minutes."""
    parts = time_str.split(':')
    hours = int(parts[0])
    minutes = int(parts[1])
    return hours * 60 + minutes

df['time_minutes'] = df['time'].apply(time_to_minutes)
print(f"   ✓ Converted time to minutes from start")

# Sort by patient and time to ensure proper ordering
df = df.sort_values(['patient_id', 'time_minutes']).reset_index(drop=True)
print(f"   ✓ Sorted data by patient_id and time")

# ============================================================================
# STEP 3: DEFINE VITAL SIGN COLUMNS
# ============================================================================
vital_sign_cols = ['HR', 'RR', 'SpO2', 'MAP']
print(f"\n3. Vital signs to process: {', '.join(vital_sign_cols)}")

# ============================================================================
# STEP 4 & 5: SLIDING WINDOW FEATURE CALCULATION
# ============================================================================
print(f"\n4. Applying sliding window feature extraction...")
print(f"   - Window size: {WINDOW_SIZE} minutes")
print(f"   - Step size: {STEP_SIZE} minute(s)")

def calculate_trend(series):
    """Calculate trend as last value minus first value."""
    if len(series) < 2:
        return np.nan
    return series.iloc[-1] - series.iloc[0]

# Store features for all patients
all_features = []

# Process each patient separately
for patient_id in df['patient_id'].unique():
    patient_data = df[df['patient_id'] == patient_id].copy()
    
    # Get demographics (same for all rows of this patient)
    age = patient_data['age'].iloc[0]
    sex = patient_data['sex'].iloc[0]
    bmi = patient_data['bmi'].iloc[0]
    
    # Calculate rolling statistics for each vital sign
    rolling_features = pd.DataFrame()
    rolling_features['patient_id'] = patient_data['patient_id']
    rolling_features['time'] = patient_data['time']
    rolling_features['time_minutes'] = patient_data['time_minutes']
    
    for vital in vital_sign_cols:
        # Create rolling window
        rolling_window = patient_data[vital].rolling(window=WINDOW_SIZE, min_periods=WINDOW_SIZE)
        
        # Calculate statistics
        rolling_features[f'{vital}_mean'] = rolling_window.mean()
        rolling_features[f'{vital}_median'] = rolling_window.median()
        rolling_features[f'{vital}_std'] = rolling_window.std()
        rolling_features[f'{vital}_min'] = rolling_window.min()
        rolling_features[f'{vital}_max'] = rolling_window.max()
        
        # Calculate trend (last - first in window)
        rolling_features[f'{vital}_trend'] = rolling_window.apply(calculate_trend, raw=False)
    
    # Add demographics to each row
    rolling_features['age'] = age
    rolling_features['sex'] = sex
    rolling_features['bmi'] = bmi
    
    # Apply step size: keep every STEP_SIZE-th row starting from first complete window
    # First complete window ends at index WINDOW_SIZE-1
    valid_indices = range(WINDOW_SIZE - 1, len(rolling_features), STEP_SIZE)
    rolling_features = rolling_features.iloc[list(valid_indices)]
    
    all_features.append(rolling_features)
    
    print(f"   ✓ Processed {patient_id}: {len(rolling_features)} feature windows")

# ============================================================================
# STEP 6: COMBINE FEATURES FROM ALL PATIENTS
# ============================================================================
print(f"\n5. Combining features from all patients...")
features_df = pd.concat(all_features, ignore_index=True)
print(f"   ✓ Combined features: {len(features_df)} total windows")

# Rename time column to window_end_time for clarity
features_df.rename(columns={'time': 'window_end_time'}, inplace=True)

# ============================================================================
# STEP 7: HANDLE NaNs FROM ROLLING (should already be handled by min_periods)
# ============================================================================
print(f"\n6. Checking for NaN values...")
nan_counts = features_df.isna().sum()
nan_cols = nan_counts[nan_counts > 0]

if len(nan_cols) > 0:
    print(f"   ! Found NaN values in columns:")
    for col, count in nan_cols.items():
        print(f"     - {col}: {count} NaNs")
    
    print(f"   - Dropping rows with NaN values...")
    before_drop = len(features_df)
    features_df = features_df.dropna()
    after_drop = len(features_df)
    print(f"   ✓ Dropped {before_drop - after_drop} rows with NaN values")
else:
    print(f"   ✓ No NaN values found")

# ============================================================================
# STEP 8: OPTIONAL PREPROCESSING
# ============================================================================
print(f"\n7. Applying optional preprocessing...")

# Get feature column names (exclude ID, time, and demographic columns)
feature_cols = [col for col in features_df.columns if any(
    col.startswith(f'{vital}_') for vital in vital_sign_cols
)]

# 8a. Encode Categorical Features
if ENCODE_CATEGORICAL:
    print(f"   - Encoding categorical variable: sex")
    # Use simple binary encoding: M=1, F=0
    features_df['sex_encoded'] = (features_df['sex'] == 'M').astype(int)
    print(f"     ✓ Encoded 'sex' as 'sex_encoded' (M=1, F=0)")
    
    # Keep original sex column for reference, but use sex_encoded for modeling
else:
    print(f"   - Skipping categorical encoding")

# 8b. Scale Numerical Features
if APPLY_SCALING:
    print(f"   - Applying StandardScaler to numerical features...")
    
    # Combine feature columns with numerical demographics for scaling
    cols_to_scale = feature_cols + ['age', 'bmi']
    
    # Initialize scaler
    scaler = StandardScaler()
    
    # Fit and transform
    features_df[cols_to_scale] = scaler.fit_transform(features_df[cols_to_scale])
    
    print(f"     ✓ Scaled {len(cols_to_scale)} numerical features")
    print(f"     ✓ Features have mean ≈ 0 and std ≈ 1")
    
    # Save scaler for future use (optional)
    import joblib
    scaler_path = os.path.join(data_dir, "feature_scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"     ✓ Saved scaler to {scaler_path}")
else:
    print(f"   - Skipping feature scaling")

# ============================================================================
# REORDER COLUMNS FOR CLARITY
# ============================================================================
print(f"\n8. Organizing columns...")

# Define column order: IDs, demographics, then features
id_cols = ['patient_id', 'window_end_time', 'time_minutes']
demo_cols = ['age', 'sex', 'bmi']
if ENCODE_CATEGORICAL:
    demo_cols.append('sex_encoded')

# Organize features by vital sign
organized_features = []
for vital in vital_sign_cols:
    vital_features = [col for col in feature_cols if col.startswith(f'{vital}_')]
    organized_features.extend(sorted(vital_features))

# Final column order
final_columns = id_cols + demo_cols + organized_features

# Reorder
features_df = features_df[final_columns]
print(f"   ✓ Organized {len(final_columns)} columns")

# ============================================================================
# STEP 9: SAVE PREPROCESSED FEATURES
# ============================================================================
print(f"\n{'='*70}")
print("SAVING PREPROCESSED FEATURES")
print('='*70)

features_df.to_csv(output_path, index=False)

print(f"\n✓ Saved: {output_path}")
print(f"✓ Total feature windows: {len(features_df)}")
print(f"✓ Total patients: {features_df['patient_id'].nunique()}")
print(f"✓ Total features per window: {len(organized_features)}")
print(f"✓ Window size: {WINDOW_SIZE} minutes")
print(f"✓ Step size: {STEP_SIZE} minute(s)")

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================
print(f"\n{'='*70}")
print("FEATURE SUMMARY")
print('='*70)

print(f"\nFeature windows per patient:")
windows_per_patient = features_df.groupby('patient_id').size()
print(f"  Mean: {windows_per_patient.mean():.1f}")
print(f"  Median: {windows_per_patient.median():.1f}")
print(f"  Min: {windows_per_patient.min()}")
print(f"  Max: {windows_per_patient.max()}")

print(f"\nFeature categories:")
for vital in vital_sign_cols:
    vital_features = [col for col in feature_cols if col.startswith(f'{vital}_')]
    print(f"  {vital}: {len(vital_features)} features ({', '.join([f.split('_')[1] for f in vital_features])})")

print(f"\nDemographic features:")
print(f"  age, sex, bmi" + (", sex_encoded" if ENCODE_CATEGORICAL else ""))

# ============================================================================
# SAMPLE DATA
# ============================================================================
print(f"\n{'='*70}")
print("SAMPLE FEATURES (First 5 rows)")
print('='*70)
# Show subset of columns for readability
sample_cols = ['patient_id', 'window_end_time', 'age', 'sex', 'bmi'] + feature_cols[:8]
print(features_df[sample_cols].head().to_string(index=False))

print(f"\n{'='*70}")
print("SAMPLE FEATURES (Last 5 rows)")
print('='*70)
print(features_df[sample_cols].tail().to_string(index=False))

print(f"\n{'='*70}")
print("✓ Feature engineering complete! Ready for model training.")
print('='*70)