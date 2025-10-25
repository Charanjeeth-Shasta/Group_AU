import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from scipy import stats, signal
from scipy.stats import linregress
import warnings
warnings.filterwarnings('ignore')

# ============ CONFIGURATION ============
WINDOW_SIZE = 30                        # Window size in minutes (rows)
STEP_SIZE = 1                           # Step size in minutes (rows)
INPUT_FILENAME = "patient_data.csv"
OUTPUT_FILENAME = "patient_features.csv"
APPLY_SCALING = True                    # Apply StandardScaler to numeric features
ENCODE_CATEGORICAL = True               # Encode categorical variables (sex)

# Normal ranges for vital signs (for time-since features)
NORMAL_RANGES = {
    'HR': (60, 100),
    'MAP': (70, 100),
    'SpO2': (95, 100),
    'RR': (12, 20)
}
# =======================================

# Setup paths
project_root = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(project_root)
input_path = os.path.join(data_dir, INPUT_FILENAME)
output_path = os.path.join(data_dir, OUTPUT_FILENAME)

print("="*70)
print("ADVANCED ICU VITAL SIGNS FEATURE ENGINEERING")
print("="*70)

# ============================================================================
# HELPER FUNCTIONS FOR ADVANCED FEATURES
# ============================================================================

def calculate_linear_slope(series):
    """
    Calculate slope of linear regression for a time series.
    Returns slope (change per time unit).
    """
    if len(series) < 2 or series.isna().all():
        return np.nan
    
    valid_idx = ~series.isna()
    if valid_idx.sum() < 2:
        return np.nan
    
    x = np.arange(len(series))[valid_idx]
    y = series[valid_idx].values
    
    try:
        slope, _, _, _, _ = linregress(x, y)
        return slope
    except:
        return np.nan


def calculate_poincare_metrics(series):
    """
    Calculate Poincaré plot metrics (SD1, SD2) for heart rate variability.
    SD1: Standard deviation perpendicular to line of identity (short-term variability)
    SD2: Standard deviation along line of identity (long-term variability)
    """
    if len(series) < 3 or series.isna().any():
        return np.nan, np.nan
    
    # Calculate successive differences
    rr_intervals = series.values
    rr_x = rr_intervals[:-1]
    rr_y = rr_intervals[1:]
    
    # Calculate SD1 and SD2
    diff_rr = rr_y - rr_x
    sum_rr = rr_y + rr_x
    
    sd1 = np.sqrt(np.var(diff_rr) / 2.0)
    sd2 = np.sqrt(2 * np.var(rr_intervals) - np.var(diff_rr) / 2.0)
    
    return sd1, sd2


def calculate_rmssd(series):
    """
    Calculate Root Mean Square of Successive Differences (RMSSD).
    Measure of heart rate variability.
    """
    if len(series) < 2 or series.isna().any():
        return np.nan
    
    diff = np.diff(series.values)
    rmssd = np.sqrt(np.mean(diff ** 2))
    
    return rmssd


def calculate_coefficient_of_variation(series):
    """
    Calculate coefficient of variation (CV = std / mean).
    Normalized measure of dispersion.
    """
    if len(series) < 2 or series.isna().all():
        return np.nan
    
    mean_val = series.mean()
    if mean_val == 0 or np.isnan(mean_val):
        return np.nan
    
    cv = series.std() / mean_val
    return cv


def time_since_normal(series, normal_range):
    """
    Calculate time (in minutes) since vital sign was last in normal range.
    Returns time since last normal value, or window size if always abnormal.
    """
    min_val, max_val = normal_range
    in_range = (series >= min_val) & (series <= max_val)
    
    if in_range.iloc[-1]:  # Current value is normal
        return 0
    
    # Find last occurrence of normal value
    last_normal_idx = in_range[::-1].idxmax() if in_range.any() else None
    
    if last_normal_idx is None:
        return len(series)  # Never in normal range during window
    
    time_since = len(series) - 1 - in_range.index.get_loc(last_normal_idx)
    return time_since


def time_since_significant_change(series, threshold=0.20):
    """
    Calculate time since last significant change (>threshold% from baseline).
    Baseline is the first value in the window.
    """
    if len(series) < 2 or series.isna().all():
        return np.nan
    
    baseline = series.iloc[0]
    if baseline == 0 or np.isnan(baseline):
        return np.nan
    
    pct_change = np.abs((series - baseline) / baseline)
    significant_change = pct_change > threshold
    
    if not significant_change.any():
        return len(series)  # No significant change in window
    
    last_change_idx = significant_change[::-1].idxmax()
    time_since = len(series) - 1 - significant_change.index.get_loc(last_change_idx)
    
    return time_since


def calculate_cross_correlation(series1, series2, max_lag=5):
    """
    Calculate maximum cross-correlation between two signals within max_lag.
    Returns max correlation coefficient and the lag at which it occurs.
    """
    if len(series1) < 3 or len(series2) < 3:
        return np.nan, np.nan
    
    if series1.isna().any() or series2.isna().any():
        return np.nan, np.nan
    
    # Normalize series
    s1 = (series1 - series1.mean()) / series1.std()
    s2 = (series2 - series2.mean()) / series2.std()
    
    correlations = []
    lags = range(-max_lag, max_lag + 1)
    
    for lag in lags:
        if lag < 0:
            corr = np.corrcoef(s1[:lag], s2[-lag:])[0, 1]
        elif lag > 0:
            corr = np.corrcoef(s1[lag:], s2[:-lag])[0, 1]
        else:
            corr = np.corrcoef(s1, s2)[0, 1]
        
        correlations.append(corr if not np.isnan(corr) else 0)
    
    max_corr = max(correlations)
    max_lag_idx = correlations.index(max_corr)
    optimal_lag = lags[max_lag_idx]
    
    return max_corr, optimal_lag


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
# STEP 4: ADVANCED SLIDING WINDOW FEATURE CALCULATION
# ============================================================================
print(f"\n4. Applying ADVANCED sliding window feature extraction...")
print(f"   - Window size: {WINDOW_SIZE} minutes")
print(f"   - Step size: {STEP_SIZE} minute(s)")
print(f"   - Basic statistics (mean, median, std, min, max, trend)")
print(f"   - Advanced trend features (linear regression slope)")
print(f"   - Interaction features (Shock Index, HR*MAP, SpO2/RR)")
print(f"   - Variability features (Poincaré, RMSSD, CV)")
print(f"   - Time-since features (normal range, significant change)")
print(f"   - Cross-signal features (correlations, phase differences)")

all_features = []
total_patients = df['patient_id'].nunique()
processed = 0

# Process each patient separately
for patient_id in df['patient_id'].unique():
    processed += 1
    patient_data = df[df['patient_id'] == patient_id].copy().reset_index(drop=True)
    
    # Get demographics
    age = patient_data['age'].iloc[0]
    sex = patient_data['sex'].iloc[0]
    bmi = patient_data['bmi'].iloc[0]
    
    # Initialize feature dataframe
    rolling_features = pd.DataFrame()
    rolling_features['patient_id'] = patient_data['patient_id']
    rolling_features['time'] = patient_data['time']
    rolling_features['time_minutes'] = patient_data['time_minutes']
    
    # ========================================================================
    # BASIC ROLLING STATISTICS
    # ========================================================================
    for vital in vital_sign_cols:
        rolling_window = patient_data[vital].rolling(window=WINDOW_SIZE, min_periods=WINDOW_SIZE)
        
        rolling_features[f'{vital}_mean'] = rolling_window.mean()
        rolling_features[f'{vital}_median'] = rolling_window.median()
        rolling_features[f'{vital}_std'] = rolling_window.std()
        rolling_features[f'{vital}_min'] = rolling_window.min()
        rolling_features[f'{vital}_max'] = rolling_window.max()
        rolling_features[f'{vital}_range'] = rolling_features[f'{vital}_max'] - rolling_features[f'{vital}_min']
        
        # Simple trend (last - first)
        rolling_features[f'{vital}_trend'] = rolling_window.apply(
            lambda x: x.iloc[-1] - x.iloc[0] if len(x) >= 2 else np.nan, raw=False
        )
    
    # ========================================================================
    # 1. ADVANCED TREND FEATURES: Linear Regression Slope
    # ========================================================================
    for vital in vital_sign_cols:
        rolling_window = patient_data[vital].rolling(window=WINDOW_SIZE, min_periods=WINDOW_SIZE)
        rolling_features[f'{vital}_slope'] = rolling_window.apply(calculate_linear_slope, raw=False)
    
    # ========================================================================
    # 2. INTERACTION FEATURES
    # ========================================================================
    # Shock Index: HR / MAP (approximation, usually HR / SBP)
    si_series = patient_data['HR'] / patient_data['MAP'].replace(0, np.nan)
    rolling_features['SI_mean'] = si_series.rolling(window=WINDOW_SIZE, min_periods=WINDOW_SIZE).mean()
    rolling_features['SI_max'] = si_series.rolling(window=WINDOW_SIZE, min_periods=WINDOW_SIZE).max()
    
    # HR * MAP product (cardiac workload proxy)
    hr_map_product = patient_data['HR'] * patient_data['MAP']
    rolling_features['HR_MAP_product_mean'] = hr_map_product.rolling(window=WINDOW_SIZE, min_periods=WINDOW_SIZE).mean()
    
    # SpO2 / RR ratio (oxygenation efficiency)
    spo2_rr_ratio = patient_data['SpO2'] / patient_data['RR'].replace(0, np.nan)
    rolling_features['SpO2_RR_ratio_mean'] = spo2_rr_ratio.rolling(window=WINDOW_SIZE, min_periods=WINDOW_SIZE).mean()
    rolling_features['SpO2_RR_ratio_min'] = spo2_rr_ratio.rolling(window=WINDOW_SIZE, min_periods=WINDOW_SIZE).min()
    
    # ========================================================================
    # 3. VARIABILITY FEATURES
    # ========================================================================
    # Poincaré metrics for HR (heart rate variability)
    hr_rolling = patient_data['HR'].rolling(window=WINDOW_SIZE, min_periods=WINDOW_SIZE)
    poincare_results = hr_rolling.apply(
        lambda x: calculate_poincare_metrics(x)[0] if len(x) >= 3 else np.nan, raw=False
    )
    rolling_features['HR_poincare_SD1'] = poincare_results
    
    poincare_results2 = hr_rolling.apply(
        lambda x: calculate_poincare_metrics(x)[1] if len(x) >= 3 else np.nan, raw=False
    )
    rolling_features['HR_poincare_SD2'] = poincare_results2
    
    # RMSSD for HR
    rolling_features['HR_rmssd'] = hr_rolling.apply(calculate_rmssd, raw=False)
    
    # Coefficient of Variation for all vitals
    for vital in vital_sign_cols:
        rolling_window = patient_data[vital].rolling(window=WINDOW_SIZE, min_periods=WINDOW_SIZE)
        rolling_features[f'{vital}_cv'] = rolling_window.apply(calculate_coefficient_of_variation, raw=False)
    
    # ========================================================================
    # 4. TIME-SINCE FEATURES
    # ========================================================================
    # Time since normal range for each vital
    for vital in vital_sign_cols:
        if vital in NORMAL_RANGES:
            rolling_window = patient_data[vital].rolling(window=WINDOW_SIZE, min_periods=WINDOW_SIZE)
            rolling_features[f'{vital}_time_since_normal'] = rolling_window.apply(
                lambda x: time_since_normal(x, NORMAL_RANGES[vital]), raw=False
            )
    
    # Time since significant change (>20% from window start)
    for vital in vital_sign_cols:
        rolling_window = patient_data[vital].rolling(window=WINDOW_SIZE, min_periods=WINDOW_SIZE)
        rolling_features[f'{vital}_time_since_change'] = rolling_window.apply(
            lambda x: time_since_significant_change(x, threshold=0.20), raw=False
        )
    
    # ========================================================================
    # 5. CROSS-SIGNAL FEATURES
    # ========================================================================
    # Pearson correlation between vital pairs in rolling window
    vital_pairs = [
        ('HR', 'MAP'),
        ('HR', 'SpO2'),
        ('RR', 'SpO2'),
        ('HR', 'RR')
    ]
    
    for v1, v2 in vital_pairs:
        corr_values = []
        for i in range(len(patient_data)):
            if i < WINDOW_SIZE - 1:
                corr_values.append(np.nan)
            else:
                window_v1 = patient_data[v1].iloc[i-WINDOW_SIZE+1:i+1]
                window_v2 = patient_data[v2].iloc[i-WINDOW_SIZE+1:i+1]
                
                if len(window_v1) >= 3 and not window_v1.isna().any() and not window_v2.isna().any():
                    corr = np.corrcoef(window_v1, window_v2)[0, 1]
                    corr_values.append(corr if not np.isnan(corr) else 0)
                else:
                    corr_values.append(np.nan)
        
        rolling_features[f'corr_{v1}_{v2}'] = corr_values
    
    # ========================================================================
    # DEVIATION FEATURES (from mean/median)
    # ========================================================================
    for vital in vital_sign_cols:
        current_value = patient_data[vital]
        rolling_features[f'{vital}_diff_from_mean'] = (
            current_value.values - rolling_features[f'{vital}_mean'].values
        )
        rolling_features[f'{vital}_diff_from_median'] = (
            current_value.values - rolling_features[f'{vital}_median'].values
        )
    
    # ========================================================================
    # ADD DEMOGRAPHICS
    # ========================================================================
    rolling_features['age'] = age
    rolling_features['sex'] = sex
    rolling_features['bmi'] = bmi
    
    # Apply step size: keep every STEP_SIZE-th row starting from first complete window
    valid_indices = range(WINDOW_SIZE - 1, len(rolling_features), STEP_SIZE)
    rolling_features = rolling_features.iloc[list(valid_indices)]
    
    all_features.append(rolling_features)
    
    print(f"   ✓ [{processed}/{total_patients}] Processed {patient_id}: {len(rolling_features)} feature windows")

# ============================================================================
# STEP 5: COMBINE FEATURES FROM ALL PATIENTS
# ============================================================================
print(f"\n5. Combining features from all patients...")
features_df = pd.concat(all_features, ignore_index=True)
print(f"   ✓ Combined features: {len(features_df)} total windows")

# Rename time column to window_end_time for clarity
features_df.rename(columns={'time': 'window_end_time'}, inplace=True)

# ============================================================================
# STEP 6: HANDLE NaNs
# ============================================================================
print(f"\n6. Checking for NaN values...")
nan_counts = features_df.isna().sum()
nan_cols = nan_counts[nan_counts > 0]

if len(nan_cols) > 0:
    print(f"   ! Found NaN values in {len(nan_cols)} columns")
    print(f"   Top columns with NaNs:")
    for col, count in nan_cols.nlargest(10).items():
        print(f"     - {col}: {count} NaNs ({count/len(features_df)*100:.1f}%)")
    
    print(f"   - Dropping rows with NaN values...")
    before_drop = len(features_df)
    features_df = features_df.dropna()
    after_drop = len(features_df)
    print(f"   ✓ Dropped {before_drop - after_drop} rows ({(before_drop-after_drop)/before_drop*100:.1f}%)")
else:
    print(f"   ✓ No NaN values found")

# ============================================================================
# STEP 7: PREPROCESSING
# ============================================================================
print(f"\n7. Applying preprocessing...")

# Get all feature column names (exclude ID, time, demographics)
feature_cols = [col for col in features_df.columns 
                if col not in ['patient_id', 'window_end_time', 'time_minutes', 
                              'age', 'sex', 'bmi', 'sex_encoded']]

print(f"   Total features extracted: {len(feature_cols)}")

# 7a. Encode Categorical Features
if ENCODE_CATEGORICAL:
    print(f"   - Encoding categorical variable: sex")
    features_df['sex_encoded'] = (features_df['sex'] == 'M').astype(int)
    print(f"     ✓ Encoded 'sex' as 'sex_encoded' (M=1, F=0)")

# 7b. Scale Numerical Features
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
    
    # Save scaler for future use
    import joblib
    scaler_path = os.path.join(data_dir, "feature_scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"     ✓ Saved scaler to {scaler_path}")

# ============================================================================
# STEP 8: ORGANIZE COLUMNS
# ============================================================================
print(f"\n8. Organizing columns...")

# Define column order: IDs, demographics, then features
id_cols = ['patient_id', 'window_end_time', 'time_minutes']
demo_cols = ['age', 'sex', 'bmi']
if ENCODE_CATEGORICAL:
    demo_cols.append('sex_encoded')

# Organize features by category
basic_features = [col for col in feature_cols if any(
    col.endswith(suffix) for suffix in ['_mean', '_median', '_std', '_min', '_max', '_range', '_trend']
)]

trend_features = [col for col in feature_cols if '_slope' in col]

interaction_features = [col for col in feature_cols if any(
    keyword in col for keyword in ['SI_', 'HR_MAP', 'SpO2_RR']
)]

variability_features = [col for col in feature_cols if any(
    keyword in col for keyword in ['poincare', 'rmssd', '_cv']
)]

time_features = [col for col in feature_cols if any(
    keyword in col for keyword in ['time_since']
)]

cross_signal_features = [col for col in feature_cols if col.startswith('corr_')]

deviation_features = [col for col in feature_cols if any(
    col.endswith(suffix) for suffix in ['_diff_from_mean', '_diff_from_median']
)]

# Combine in logical order
organized_features = (basic_features + trend_features + interaction_features + 
                     variability_features + time_features + cross_signal_features + 
                     deviation_features)

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
# FEATURE SUMMARY
# ============================================================================
print(f"\n{'='*70}")
print("ADVANCED FEATURE SUMMARY")
print('='*70)

print(f"\nFeature windows per patient:")
windows_per_patient = features_df.groupby('patient_id').size()
print(f"  Mean: {windows_per_patient.mean():.1f}")
print(f"  Median: {windows_per_patient.median():.1f}")
print(f"  Min: {windows_per_patient.min()}")
print(f"  Max: {windows_per_patient.max()}")

print(f"\nFeature categories:")
print(f"  Basic statistics: {len(basic_features)} features")
print(f"  Trend features (slopes): {len(trend_features)} features")
print(f"  Interaction features: {len(interaction_features)} features")
print(f"  Variability features: {len(variability_features)} features")
print(f"  Time-since features: {len(time_features)} features")
print(f"  Cross-signal features: {len(cross_signal_features)} features")
print(f"  Deviation features: {len(deviation_features)} features")
print(f"  ---")
print(f"  TOTAL: {len(organized_features)} features")

print(f"\nKey clinical features:")
print(f"  ✓ Shock Index (HR/MAP) - mean and max")
print(f"  ✓ HR*MAP product (cardiac workload)")
print(f"  ✓ SpO2/RR ratio (oxygenation efficiency)")
print(f"  ✓ Heart Rate Variability (Poincaré SD1/SD2, RMSSD)")
print(f"  ✓ Coefficient of Variation for all vitals")
print(f"  ✓ Time since normal range for each vital")
print(f"  ✓ Time since significant change (>20%)")
print(f"  ✓ Cross-correlations between vital pairs")
print(f"  ✓ Linear regression slopes (trend direction)")

print(f"\nDemographic features:")
print(f"  age, sex, bmi" + (", sex_encoded" if ENCODE_CATEGORICAL else ""))

# ============================================================================
# SAMPLE DATA
# ============================================================================
print(f"\n{'='*70}")
print("SAMPLE FEATURES (First 3 rows)")
print('='*70)

# Show subset of columns for readability
sample_cols = ['patient_id', 'window_end_time', 'age', 'sex'] + organized_features[:12]
print(features_df[sample_cols].head(3).to_string(index=False))

print(f"\n{'='*70}")
print("ADVANCED FEATURES PREVIEW")
print('='*70)

# Show key advanced features
advanced_preview = ['patient_id', 'window_end_time',
                   'HR_slope', 'MAP_slope', 'SI_mean', 'HR_MAP_product_mean',
                   'HR_poincare_SD1', 'HR_rmssd', 'HR_time_since_normal',
                   'corr_HR_MAP', 'SpO2_RR_ratio_mean']
available_preview = [col for col in advanced_preview if col in features_df.columns]
print(features_df[available_preview].head(3).to_string(index=False))

print(f"\n{'='*70}")
print("✓ ADVANCED FEATURE ENGINEERING COMPLETE!")
print("✓ Ready for model training with enhanced predictive features")
print('='*70)