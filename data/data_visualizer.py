# -*- coding: utf-8 -*-
"""
Plots vital signs (HR, RR, SpO2, MAP) from patient_sim.csv
and highlights outliers using the IQR method, with improved plot formatting.
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates # Import module for date formatting
import numpy as np

# --- Configuration ---
CSV_FILE = 'patient_sim.csv' # Make sure this file is in the same directory

# Define the exact column names expected in the CSV header
TIMESTAMP_COLUMN_NAME = 'time' # Column name for the timestamp
VITAL_COLUMNS_TO_PLOT = ['HR', 'RR', 'SpO2', 'MAP'] # Exact column names for vitals
# Define the expected time format in the CSV
TIME_FORMAT = '%H:%M:%S'

# --- Function to Find Outliers using IQR ---
def find_outliers_iqr(series):
    """Identifies outliers in a pandas Series using the IQR method."""
    numeric_series = pd.to_numeric(series, errors='coerce')
    q1 = numeric_series.quantile(0.25)
    q3 = numeric_series.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    if pd.isna(lower_bound) or pd.isna(upper_bound):
        return pd.Series([False] * len(series), index=series.index)
    return (numeric_series < lower_bound) | (numeric_series > upper_bound)

# --- 1. Read and Prepare Data ---
try:
    print(f"Reading '{CSV_FILE}' assuming it has a header...")
    df = pd.read_csv(CSV_FILE)

    print("\nFirst 5 rows as read:")
    print(df.head())
    print("\nDataFrame columns found:", df.columns.tolist())

    # --- Check and Convert Timestamp Column ---
    if TIMESTAMP_COLUMN_NAME not in df.columns:
        print(f"\nError: Timestamp column '{TIMESTAMP_COLUMN_NAME}' not found.")
        exit()

    print(f"\nConverting '{TIMESTAMP_COLUMN_NAME}' column to datetime using format '{TIME_FORMAT}'...")
    # Convert time string to datetime objects based on a dummy date (for plotting time axis)
    # Using today's date, but only the time part will be used for display
    df['Timestamp_dt'] = pd.to_datetime(df[TIMESTAMP_COLUMN_NAME], format=TIME_FORMAT, errors='coerce')

    original_rows = len(df)
    failed_conversions = df['Timestamp_dt'].isna().sum()
    if failed_conversions > 0:
         print(f"\nWarning: {failed_conversions} rows had timestamps that did NOT match format '{TIME_FORMAT}'.")
         failed_examples = df[df['Timestamp_dt'].isna()][TIMESTAMP_COLUMN_NAME].unique()
         print(f"  Unique values that failed conversion (first 5): {failed_examples[:5]}")
         df = df.dropna(subset=['Timestamp_dt'])
         print(f"  Dropped {failed_conversions} rows due to invalid timestamps.")

    # --- Process Vital Signs ---
    print("\nConverting vital sign columns to numeric...")
    vitals_present = []
    for col in VITAL_COLUMNS_TO_PLOT:
        if col in df.columns:
            print(f"  Processing '{col}'...")
            df[col] = pd.to_numeric(df[col], errors='coerce')
            if df[col].isna().any():
                 nan_count = df[col].isna().sum()
                 # Count only newly introduced NaNs not already dropped by time errors
                 current_valid_indices = df[col].notna().index
                 new_nan_count = df.loc[current_valid_indices, col].isna().sum()
                 if new_nan_count > 0:
                      print(f"    Warning: Found {new_nan_count} non-numeric values in '{col}'. Coerced to NaN.")
            vitals_present.append(col)
        else:
             print(f"  Warning: Expected vital sign column '{col}' not found. Skipping.")

    VITAL_COLUMNS_TO_PLOT = vitals_present # Update list to only those found

    original_rows_before_vital_drop = len(df)
    df = df.dropna(subset=VITAL_COLUMNS_TO_PLOT)
    dropped_for_vitals = original_rows_before_vital_drop - len(df)
    if dropped_for_vitals > 0:
         print(f"\nWarning: Dropped {dropped_for_vitals} rows due to missing/non-numeric vital signs.")

    if df.empty or not VITAL_COLUMNS_TO_PLOT:
        print(f"\nError: No valid data remains after processing.")
        exit()
    else:
        print(f"\nData successfully processed. Found {len(df)} valid rows.")
        print("Columns available for plotting:", VITAL_COLUMNS_TO_PLOT)

except FileNotFoundError:
    print(f"Error: File '{CSV_FILE}' not found.")
    exit()
except ValueError as ve:
    print(f"Error processing CSV data: {ve}")
    exit()
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    exit()


# --- 2. Identify Outliers ---
outlier_indices = {}
print("\n--- Calculating Outliers ---")
for col in VITAL_COLUMNS_TO_PLOT:
    outliers_bool = find_outliers_iqr(df[col])
    outlier_indices[col] = df.index[outliers_bool].tolist()
    print(f"Checked {col}...")

# --- 3. Plotting ---
print("--- Generating Plots ---")
num_vitals = len(VITAL_COLUMNS_TO_PLOT)
fig, axes = plt.subplots(num_vitals, 1, figsize=(14, num_vitals * 3), sharex=True, squeeze=False)
axes = axes.flatten()

# --- Configure X-axis Tick Formatting ---
# Use AutoDateLocator to select good tick locations
locator = mdates.AutoDateLocator(minticks=5, maxticks=10) # Adjust minticks/maxticks as needed
# Use ConciseDateFormatter for clean time display (HH:MM)
formatter = mdates.ConciseDateFormatter(locator, formats=['%H:%M:%S', '%H:%M', '%H:%M', '%H:%M', '%H:%M', '%H:%M'], zero_formats=['', '%H:%M', '%H:%M', '%H:%M', '%H:%M', '%H:%M'])


for i, col in enumerate(VITAL_COLUMNS_TO_PLOT):
    ax = axes[i]

    # Plot the main time series data points
    ax.plot(df['Timestamp_dt'], df[col], label=f'{col} Values', marker='.', markersize=5, linestyle='-', linewidth=1.5, zorder=1)

    # Highlight the identified outliers on the plot
    if outlier_indices[col]:
        outlier_df = df.loc[outlier_indices[col]]
        ax.scatter(outlier_df['Timestamp_dt'], outlier_df[col], color='red', marker='x', s=80, label='Outliers (IQR)', zorder=2, alpha=0.9)

    # Set labels and title
    ax.set_ylabel(col)
    ax.set_title(f'{col} Time Series')
    ax.legend(loc='best')
    ax.grid(True, linestyle='--', alpha=0.7)

    # --- Apply the improved X-axis formatting ---
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

# --- Final Plot Adjustments ---
axes[-1].set_xlabel("Time") # Label only the bottom x-axis

# Rotate x-tick labels slightly if needed (often handled by ConciseDateFormatter)
# plt.setp(axes[-1].get_xticklabels(), rotation=30, ha='right')

# Adjust layout
plt.tight_layout(pad=2.0)
plt.show()

# --- Print Outlier Summary ---
print("\n--- Outlier Detection Summary ---")
# (Summary code remains the same as before)
for col in VITAL_COLUMNS_TO_PLOT:
     indices = outlier_indices[col]
     if indices:
         print(f"\n- {col}: Outliers found at DataFrame index positions: {indices}")
         print(f"  Values: {df.loc[indices, col].round(2).tolist()}")
     else:
         print(f"\n- {col}: No outliers detected using IQR method.")

print("\n--- Script Finished ---")