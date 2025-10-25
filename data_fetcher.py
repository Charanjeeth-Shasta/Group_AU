import pandas as pd
import numpy as np
import os

try:
    import vitaldb
except ImportError:
    print("Please install vitaldb library first: pip install vitaldb")
    exit()

# ============ CONFIGURATION ============
NUM_PATIENTS = 5         # Number of patients to load
MIN_DURATION_MINUTES = 120 # Minimum duration per patient
TARGET_INTERVAL = 60       # Sampling interval in seconds
# =======================================

print(f"Loading data from {NUM_PATIENTS} patients...")

# Track names to load
track_names = [
    'Solar8000/HR',
    'Solar8000/PLETH_SPO2',
    'Solar8000/RR_CO2',
    'Solar8000/ART_MBP',
]

all_patient_data = []

# Try to load real patient data
case_ids_to_try = [1, 2, 3, 4, 5, 100, 200, 300, 400, 500]

for idx, case_id in enumerate(case_ids_to_try[:NUM_PATIENTS]):
    print(f"\n--- Patient {idx + 1} (Case ID: {case_id}) ---")
    
    try:
        # Load vital signs
        vals = vitaldb.load_case(case_id, track_names, TARGET_INTERVAL)
        
        if vals is not None and len(vals) > MIN_DURATION_MINUTES:
            print(f"  ✓ Loaded {len(vals)} samples")
            
            # Create DataFrame
            patient_df = pd.DataFrame(vals, columns=['HR', 'SpO2', 'RR', 'MAP'])
            patient_df = patient_df.dropna(how='all').ffill().bfill().dropna()
            
            # Add patient ID
            patient_df.insert(0, 'patient_id', f'P{idx + 1:03d}')
            
            # Try to get patient demographics from VitalDB
            try:
                # Load case info
                vf = vitaldb.VitalFile(case_id)
                case_info = vf.get_header()
                
                # Extract demographics
                age = case_info.get('age', np.random.randint(30, 80))
                sex = case_info.get('sex', np.random.choice(['M', 'F']))
                height = case_info.get('height', np.random.randint(155, 185))
                weight = case_info.get('weight', np.random.randint(55, 95))
                
            except:
                # Generate realistic demographics if not available
                age = np.random.randint(30, 80)
                sex = np.random.choice(['M', 'F'])
                height = np.random.randint(155, 185)
                weight = np.random.randint(55, 95)
            
            # Calculate BMI
            bmi = weight / ((height / 100) ** 2)
            
            # Add demographics to dataframe
            patient_df['age'] = age
            patient_df['sex'] = sex
            patient_df['height_cm'] = height
            patient_df['weight_kg'] = weight
            patient_df['bmi'] = round(bmi, 1)
            
            all_patient_data.append(patient_df)
            print(f"  Patient: {age}yr {sex}, {height}cm, {weight}kg, BMI {bmi:.1f}")
            
        else:
            print(f"  ⚠ Insufficient data, skipping")
            
    except Exception as e:
        print(f"  ✗ Error: {e}")

# If we don't have enough patients, generate synthetic data
patients_loaded = len(all_patient_data)

if patients_loaded < NUM_PATIENTS:
    print(f"\nGenerating synthetic data for {NUM_PATIENTS - patients_loaded} additional patients...")
    
    np.random.seed(42)
    
    for idx in range(patients_loaded, NUM_PATIENTS):
        print(f"\n--- Patient {idx + 1} (Synthetic) ---")
        
        # Generate patient demographics
        age = np.random.randint(25, 85)
        sex = np.random.choice(['M', 'F'])
        
        if sex == 'M':
            height = np.random.randint(165, 190)
            weight = np.random.randint(60, 100)
        else:
            height = np.random.randint(155, 175)
            weight = np.random.randint(50, 85)
        
        bmi = weight / ((height / 100) ** 2)
        
        # Generate duration (2-6 hours)
        n_samples = np.random.randint(120, 360)
        time = np.arange(n_samples)
        
        # Add patient-specific baseline variations
        hr_baseline = 65 + (age - 50) * 0.2 + np.random.randn() * 5
        rr_baseline = 14 + (age - 50) * 0.05 + np.random.randn() * 2
        spo2_baseline = 98 - (age - 50) * 0.05
        map_baseline = 80 + (age - 50) * 0.3 + np.random.randn() * 5
        
        # Circadian and trend patterns
        circadian = 8 * np.sin(2 * np.pi * time / 360)
        
        # Heart Rate
        hr = hr_baseline + np.random.randn(n_samples) * 4 + 0.01 * time + circadian * 0.8
        hr = hr.clip(45, 140)
        
        # Respiratory Rate
        rr = rr_baseline + np.random.randn(n_samples) * 1.5 + 0.005 * time + circadian * 0.2
        rr = rr.clip(10, 30)
        
        # SpO2
        spo2 = spo2_baseline + np.random.randn(n_samples) * 0.8 - 0.003 * time
        spo2 = spo2.clip(88, 100)
        
        # MAP
        map_val = map_baseline + np.random.randn(n_samples) * 5 - 0.008 * time + circadian * 0.6
        map_val = map_val.clip(60, 120)
        
        # Create DataFrame
        patient_df = pd.DataFrame({
            'patient_id': f'P{idx + 1:03d}',
            'HR': hr,
            'RR': rr,
            'SpO2': spo2,
            'MAP': map_val,
            'age': age,
            'sex': sex,
            'height_cm': height,
            'weight_kg': weight,
            'bmi': round(bmi, 1)
        })
        
        all_patient_data.append(patient_df)
        print(f"  Generated: {age}yr {sex}, {height}cm, {weight}kg, BMI {bmi:.1f}, {n_samples} samples")

# Combine all patient data
print("\n" + "="*60)
print("COMBINING ALL PATIENT DATA")
print("="*60)

combined_df = pd.concat(all_patient_data, ignore_index=True)

print(f"✓ Total patients: {NUM_PATIENTS}")
print(f"✓ Total samples: {len(combined_df)}")

# Add relative time for each patient
time_col = []
for patient_id in combined_df['patient_id'].unique():
    patient_samples = len(combined_df[combined_df['patient_id'] == patient_id])
    patient_times = [f"{i // 60:02d}:{i % 60:02d}:00" for i in range(patient_samples)]
    time_col.extend(patient_times)

combined_df.insert(1, 'time', time_col)

# Round numeric values
numeric_cols = ['HR', 'RR', 'SpO2', 'MAP']
combined_df[numeric_cols] = combined_df[numeric_cols].round(1)

# Reorder columns for better readability
column_order = ['patient_id', 'time', 'age', 'sex', 'height_cm', 'weight_kg', 'bmi', 
                'HR', 'RR', 'SpO2', 'MAP']
combined_df = combined_df[column_order]

# Save to CSV
output_filename = '/data/patient_data.csv'
output_filename = os.path.normpath(output_filename)  # normalize Windows/Unix slashes
parent = os.path.dirname(output_filename)
if parent:
    os.makedirs(parent, exist_ok=True)  # create directory if missing

combined_df.to_csv(output_filename, index=False)

print(f"\n✓ Created {output_filename}")
print(f"  Total rows: {len(combined_df)}")
print(f"  Patients: {combined_df['patient_id'].nunique()}")
print(f"  Duration per patient: {len(combined_df) // NUM_PATIENTS} minutes avg")

# Show summary statistics by patient
print("\n" + "="*60)
print("PATIENT SUMMARY")
print("="*60)

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
    print(f"  HR: {patient_data['HR'].mean():.1f} ± {patient_data['HR'].std():.1f} bpm")
    print(f"  RR: {patient_data['RR'].mean():.1f} ± {patient_data['RR'].std():.1f} /min")
    print(f"  SpO2: {patient_data['SpO2'].mean():.1f} ± {patient_data['SpO2'].std():.1f} %")
    print(f"  MAP: {patient_data['MAP'].mean():.1f} ± {patient_data['MAP'].std():.1f} mmHg")

# Show sample data
print("\n" + "="*60)
print("SAMPLE DATA (first 10 rows)")
print("="*60)
print(combined_df.head(10).to_string(index=False))

print("\n" + "="*60)
print("SAMPLE DATA (last 10 rows)")
print("="*60)
print(combined_df.tail(10).to_string(index=False))

print("\n✓ Dataset ready for patient monitoring simulation!")