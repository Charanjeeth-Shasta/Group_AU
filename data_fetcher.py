import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

try:
    import vitaldb
except ImportError:
    print("Please install vitaldb library first: pip install vitaldb")
    exit()

# ============ CONFIGURATION ============
NUM_PATIENTS = 50                        # Number of patients to load (must be even)
TARGET_DURATION_MINUTES = 180           # Exact duration per patient 
TARGET_INTERVAL = 60                    # Sampling interval in seconds (1 minute)
MIN_DURATION_MINUTES = TARGET_DURATION_MINUTES
min_samples_per_patient = MIN_DURATION_MINUTES

# Validate configuration
if NUM_PATIENTS % 2 != 0:
    print("ERROR: NUM_PATIENTS must be even to get equal male/female split")
    exit(1)

PATIENTS_PER_GENDER = NUM_PATIENTS // 2
# =======================================

# Track names to load
track_names = [
    'Solar8000/HR',
    'Solar8000/PLETH_SPO2',
    'Solar8000/RR_CO2',
    'Solar8000/ART_MBP',
]

# Setup output directory
project_root = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(project_root, "data")
os.makedirs(data_dir, exist_ok=True)
output_filename = os.path.join(data_dir, "test_patient_data.csv")

# Replace the API URL with environment variable
VITALDB_API_URL = os.getenv('VITALDB_API_URL')
VITALDB_API_KEY = os.getenv('VITALDB_API_KEY')

# Load clinical data from VitalDB API
print("Loading clinical information from VitalDB...")
try:
    clinical_df = pd.read_csv(f'{VITALDB_API_URL}{os.getenv("VITALDB_CASES_ENDPOINT")}')
    print(f"Loaded clinical data for {len(clinical_df)} cases\n")
except Exception as e:
    print(f"Error loading clinical data: {e}")
    exit(1)

def get_case_demographics(case_id):
    """Retrieve patient demographics from clinical dataframe."""
    try:
        case_row = clinical_df[clinical_df['caseid'] == case_id]
        if len(case_row) == 0:
            return None
        
        case_row = case_row.iloc[0]
        age = case_row.get('age')
        sex = case_row.get('sex')
        height = case_row.get('height')
        weight = case_row.get('weight')
        
        if pd.isna(age) or pd.isna(sex) or pd.isna(height) or pd.isna(weight):
            return None
        if not all([age, sex, height, weight]):
            return None
        
        bmi = weight / ((height / 100) ** 2)
        
        return {
            'age': int(age),
            'sex': sex,
            'height_cm': int(height),
            'weight_kg': int(weight),
            'bmi': round(bmi, 1)
        }
    except:
        return None

def fetch_cases_page(page: int = 1):
    """Fetch a page of case IDs from VitalDB clinical data."""
    cases_per_page = 50
    start_idx = (page - 1) * cases_per_page
    end_idx = start_idx + cases_per_page
    
    all_case_ids = clinical_df['caseid'].tolist()
    page_case_ids = all_case_ids[start_idx:end_idx]
    
    cases = [{'id': case_id, 'case_id': case_id, 'caseId': case_id} for case_id in page_case_ids]
    
    return {'cases': cases, 'total': len(all_case_ids)}

def load_case_samples(case, required_gender=None):
    """Load vital signs data for a specific case."""
    case_id = case.get('id') or case.get('case_id') or case.get('caseId')
    
    try:
        # Get patient demographics first to check gender
        demographics = get_case_demographics(case_id)
        if demographics is None:
            return None
        
        # Check if we need a specific gender
        if required_gender and demographics['sex'] != required_gender:
            return None
        
        # Load vital signs
        vals = vitaldb.load_case(case_id, track_names, TARGET_INTERVAL)
        
        if vals is None or len(vals) < MIN_DURATION_MINUTES:
            return None
        
        # Truncate to exact target duration
        vals = vals[:TARGET_DURATION_MINUTES]
        
        # Create DataFrame
        patient_df = pd.DataFrame(vals, columns=['HR', 'SpO2', 'RR', 'MAP'])
        patient_df = patient_df.dropna(how='all').ffill().bfill().dropna()
        
        # Ensure exact target duration after cleaning
        if len(patient_df) < TARGET_DURATION_MINUTES:
            return None
        
        # Truncate to exact target duration
        patient_df = patient_df.head(TARGET_DURATION_MINUTES)
        
        # Add patient ID
        patient_idx = len(collected_cases) + 1
        patient_df.insert(0, 'patient_id', f'P{patient_idx:03d}')
        
        # Add demographics
        patient_df['age'] = demographics['age']
        patient_df['sex'] = demographics['sex']
        patient_df['height_cm'] = demographics['height_cm']
        patient_df['weight_kg'] = demographics['weight_kg']
        patient_df['bmi'] = demographics['bmi']
        
        # Add time column
        n_samples = len(patient_df)
        patient_times = [f"{i // 60:02d}:{i % 60:02d}:00" for i in range(n_samples)]
        patient_df.insert(1, 'time', patient_times)
        
        # Round numeric values
        numeric_cols = ['HR', 'SpO2', 'RR', 'MAP']
        patient_df[numeric_cols] = patient_df[numeric_cols].round(1)
        
        # Reorder columns
        column_order = ['patient_id', 'time', 'age', 'sex', 'height_cm', 'weight_kg', 'bmi', 
                        'HR', 'RR', 'SpO2', 'MAP']
        patient_df = patient_df[column_order]
        
        print(f"✓ Case {case_id}: {demographics['age']}yr {demographics['sex']}, "
              f"{demographics['height_cm']}cm, {demographics['weight_kg']}kg, BMI {demographics['bmi']}")
        
        return patient_df
        
    except Exception as e:
        return None

# Collection loop
collected_cases = {}
all_dfs = []
page = 1
max_pages = 100
total_cases = None

# Track gender counts
gender_counts = {'M': 0, 'F': 0}

print(f"Collecting {NUM_PATIENTS} patients ({PATIENTS_PER_GENDER} male, {PATIENTS_PER_GENDER} female) with {TARGET_DURATION_MINUTES} minutes each...\n")

while len(collected_cases) < NUM_PATIENTS:
    result = fetch_cases_page(page=page)
    
    if result is None:
        break

    if isinstance(result, dict):
        cases_page = result.get("cases") or result.get("data") or result.get("results") or []
        if total_cases is None:
            total_cases = result.get("total") or result.get("total_cases") or result.get("count")
    elif isinstance(result, tuple):
        cases_page = result[0] or []
        if total_cases is None and len(result) > 1:
            total_cases = result[1]
    elif isinstance(result, list):
        cases_page = result
    else:
        raise TypeError("Unsupported return type from fetch_cases_page.")

    if not cases_page:
        break

    for case in cases_page:
        if len(collected_cases) >= NUM_PATIENTS:
            break
            
        case_id = str(case.get("id") or case.get("case_id") or case.get("caseId") or "")
        if not case_id or case_id in collected_cases:
            continue

        # Determine which gender we need
        required_gender = None
        if gender_counts['M'] >= PATIENTS_PER_GENDER:
            required_gender = 'F'  # Need more females
        elif gender_counts['F'] >= PATIENTS_PER_GENDER:
            required_gender = 'M'  # Need more males

        df = load_case_samples(case, required_gender=required_gender)
        if df is None:
            continue

        if len(df) < min_samples_per_patient:
            continue

        # Get gender from dataframe
        patient_gender = df['sex'].iloc[0]
        
        collected_cases[case_id] = case
        all_dfs.append(df)
        gender_counts[patient_gender] += 1

    if len(collected_cases) >= NUM_PATIENTS:
        break

    page += 1
    if page > max_pages:
        break

if not collected_cases:
    raise RuntimeError("No valid patients collected.")

# Combine data
print(f"\n{'='*70}")
print("COMBINING DATA")
print('='*70)

combined_df = pd.concat(all_dfs, ignore_index=True)
print(f"✓ Combined {len(collected_cases)} patients into {len(combined_df)} total rows")

# ============================================================================
# DATA CLEANING STEPS - Applied before saving to ensure model-ready data
# ============================================================================

print(f"\n{'='*70}")
print("APPLYING DATA CLEANING STEPS")
print('='*70)

# Store original row count for reporting
original_row_count = len(combined_df)

# Step 1: Handle Invalid Placeholders
print("\n1. Handling invalid placeholders...")
print(f"   - Replacing MAP values of -9.0 with NaN")
invalid_map_count = (combined_df['MAP'] == -9.0).sum()
combined_df['MAP'] = combined_df['MAP'].replace(-9.0, np.nan)
print(f"   ✓ Replaced {invalid_map_count} invalid MAP values")

# Step 2: Ensure Numeric Data Types
print("\n2. Ensuring numeric data types...")
vital_sign_cols = ['HR', 'RR', 'SpO2', 'MAP']
for col in vital_sign_cols:
    non_numeric_count = pd.to_numeric(combined_df[col], errors='coerce').isna().sum() - combined_df[col].isna().sum()
    combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')
    if non_numeric_count > 0:
        print(f"   ✓ Converted {non_numeric_count} non-numeric values to NaN in {col}")

# Step 3: Handle Missing Values (NaN)
print("\n3. Handling missing values...")
print(f"   - Sorting by patient_id and time")
combined_df = combined_df.sort_values(by=['patient_id', 'time']).reset_index(drop=True)

print(f"   - Applying forward fill (ffill) within each patient group")
# Count NaN values before fills
nan_counts_before = combined_df[vital_sign_cols].isna().sum()

# Group by patient and apply forward fill to vital sign columns
combined_df[vital_sign_cols] = combined_df.groupby('patient_id')[vital_sign_cols].ffill()

# Count NaNs after ffill but before bfill
nan_counts_after_ffill = combined_df[vital_sign_cols].isna().sum()

print(f"   - Applying backward fill (bfill) within each patient group to fill initial NaNs")
# Group by patient and apply backward fill
combined_df[vital_sign_cols] = combined_df.groupby('patient_id')[vital_sign_cols].bfill()

# Count NaNs after bfill
nan_counts_after_bfill = combined_df[vital_sign_cols].isna().sum()

for col in vital_sign_cols:
    filled_ffill = nan_counts_before[col] - nan_counts_after_ffill[col]
    filled_bfill = nan_counts_after_ffill[col] - nan_counts_after_bfill[col]
    if filled_ffill > 0:
        print(f"   ✓ Forward filled {filled_ffill} NaN values in {col}")
    if filled_bfill > 0:
        print(f"   ✓ Backward filled {filled_bfill} initial NaN values in {col}")

# Check if any NaNs STILL remain (should only happen if a patient has NO valid data for a vital)
final_nan_count = combined_df[vital_sign_cols].isna().sum().sum()
if final_nan_count > 0:
    print(f"   ! Warning: {final_nan_count} NaNs still remain after ffill and bfill.")
    print("     This might occur if a patient has no valid data for an entire vital sign column.")
    print("     Dropping rows with persistent NaNs...")
    rows_before_drop = len(combined_df)
    combined_df = combined_df.dropna(subset=vital_sign_cols)
    rows_dropped = rows_before_drop - len(combined_df)
    print(f"     ✓ Dropped {rows_dropped} rows with persistent NaNs.")

# Step 4: Handle Physiological Outliers
print("\n4. Applying physiological limits...")
outlier_limits = {
    'HR': (30, 220),
    'RR': (3, 60),
    'SpO2': (70, 100),
    'MAP': (40, 160)
}

for col, (min_val, max_val) in outlier_limits.items():
    outliers_low = (combined_df[col] < min_val).sum()
    outliers_high = (combined_df[col] > max_val).sum()
    combined_df[col] = combined_df[col].clip(lower=min_val, upper=max_val)
    
    total_outliers = outliers_low + outliers_high
    if total_outliers > 0:
        print(f"   ✓ {col}: Clipped {total_outliers} outliers ({outliers_low} below {min_val}, {outliers_high} above {max_val})")

# Round vital signs to 1 decimal place after cleaning
combined_df[vital_sign_cols] = combined_df[vital_sign_cols].round(1)

print(f"\n{'='*70}")
print("CLEANING SUMMARY")
print('='*70)
print(f"Original rows: {original_row_count}")
print(f"Final rows: {len(combined_df)}")
print(f"Rows removed: {original_row_count - len(combined_df)}")
print(f"Data retention: {100 * len(combined_df) / original_row_count:.1f}%")

# ============================================================================
# SAVE CLEANED DATA
# ============================================================================

print(f"\n{'='*70}")
print("SAVING CLEANED DATA")
print('='*70)

combined_df.to_csv(output_filename, index=False)

print(f"\n✓ Saved: {output_filename}")
print(f"✓ Total rows: {len(combined_df)}")
print(f"✓ Patients: {len(collected_cases)} (Male: {gender_counts['M']}, Female: {gender_counts['F']})")
print(f"✓ Duration per patient: {TARGET_DURATION_MINUTES} minutes")

# Summary statistics
print(f"\n{'='*70}")
print("PATIENT SUMMARY")
print('='*70)

for patient_id in combined_df['patient_id'].unique():
    patient_data = combined_df[combined_df['patient_id'] == patient_id]
    age = patient_data['age'].iloc[0]
    sex = patient_data['sex'].iloc[0]
    height = patient_data['height_cm'].iloc[0]
    weight = patient_data['weight_kg'].iloc[0]
    bmi = patient_data['bmi'].iloc[0]
    duration = len(patient_data)
    
    print(f"\n{patient_id}: {age}yr {sex}, {height}cm, {weight}kg, BMI {bmi}")
    print(f"  Duration: {duration} minutes ({duration/60:.1f} hours)")
    print(f"  HR:   {patient_data['HR'].mean():.1f} ± {patient_data['HR'].std():.1f} bpm")
    print(f"  RR:   {patient_data['RR'].mean():.1f} ± {patient_data['RR'].std():.1f} /min")
    print(f"  SpO2: {patient_data['SpO2'].mean():.1f} ± {patient_data['SpO2'].std():.1f} %")
    print(f"  MAP:  {patient_data['MAP'].mean():.1f} ± {patient_data['MAP'].std():.1f} mmHg")

# Sample data
print(f"\n{'='*70}")
print("SAMPLE DATA (First 10 rows)")
print('='*70)
print(combined_df.head(10).to_string(index=False))

print(f"\n{'='*70}")
print("SAMPLE DATA (Last 10 rows)")
print('='*70)
print(combined_df.tail(10).to_string(index=False))

print(f"\n{'='*70}")
print(f"✓ Dataset ready! Case IDs: {list(collected_cases.keys())}")
print('='*70)