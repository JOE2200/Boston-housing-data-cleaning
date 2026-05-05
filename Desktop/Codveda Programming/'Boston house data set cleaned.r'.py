# Task 1: Data Cleaning and Preprocessing
# Boston Housing Dataset

import pandas as pd
import numpy as np

print("=" * 40)
print("TASK 1: DATA CLEANING AND PREPROCESSING")
print("=" * 40)

# Load my dataset with the rule of thumb column names
print("\n1. LOADING THE DATASET")
print("-" * 40)

# showing the column names for Boston Housing dataset
column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD',
                'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']  
file_name = 'Boston house data set cleaned.r'  
try:
    data = pd.read_csv(file_name, delim_whitespace=True, names=column_names)
    print(f"✓ Successfully loaded: {file_name}")
    print(f"  Dataset shape: {data.shape[0]} rows × {data.shape[1]} columns")
    print(f"  Column names: {list(data.columns)}")
except FileNotFoundError:
    print(f"✗ File not found: {file_name}")
    print("  Please make sure the file is in the current directory.")
    exit()

# Here I am performing Data Quality Assessment
print("\n2. DATA QUALITY ASSESSMENT")
print("=" * 60)

print(f"\nNumber of rows and columns: {data.shape}")
print(f"Total data points: {data.size}")

# Check for any missing values
print("\n3. CHECKING FOR MISSING VALUES")
print("=" * 60)

missing_count = data.isnull().sum()
print("\nNumber of missing values in each column:")
print(missing_count)

print("\nPercentage of missing values in each column:")
missing_percentage = (missing_count / len(data)) * 100
print(missing_percentage.round(2))

if missing_count.sum() > 0:
    print(f"\n⚠ Total missing values: {missing_count.sum()}")
    print(f"  Columns with missing values: {missing_count[missing_count > 0].index.tolist()}")
else:
    print("\n✓ No missing values found in the dataset!")

# Handle missing values if any is found
print("\n4. HANDLING MISSING VALUES")
print("=" * 60)

if missing_count.sum() > 0:
    print("Handling missing values...")
    for column in data.columns:
        if data[column].isnull().sum() > 0:
            if data[column].dtype in ['int64', 'float64']:
                median_val = data[column].median()
                data[column].fillna(median_val, inplace=True)
                print(f"  ✓ {column}: filled with median ({median_val:.2f})")
            else:
                mode_val = data[column].mode()[0]
                data[column].fillna(mode_val, inplace=True)
                print(f"  ✓ {column}: filled with mode ({mode_val})")
    print("\n✓ All missing values handled!")
else:
    print("✓ No missing values to handle!")

# Detect and remove duplicate rows
print("\n5. DETECTING AND REMOVING DUPLICATE ROWS")
print("=" * 60)

duplicate_count = data.duplicated().sum()
print(f"Number of duplicate rows: {duplicate_count}")

if duplicate_count > 0:
    print(f"Percentage of duplicate rows: {(duplicate_count / len(data)) * 100:.2f}%")
    data = data.drop_duplicates()
    print(f"✓ Removed {duplicate_count} duplicate rows")
    print(f"  New dataset shape: {data.shape}")
else:
    print("✓ No duplicate rows found!")

# This is to find and handle outliers
print("\n6. OUTLIER DETECTION AND HANDLING")
print("=" * 60)

# Method: "cap" or "remove"
outlier_method = "cap"

for column in data.columns:
    if data[column].dtype in ['int64', 'float64']:
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
        outlier_count = len(outliers)
        
        if outlier_count > 0:
            print(f"\n{column}:")
            print(f"  Outliers detected: {outlier_count} ({outlier_count/len(data)*100:.1f}%)")
            print(f"  Lower bound: {lower_bound:.2f}")
            print(f"  Upper bound: {upper_bound:.2f}")
            
            if outlier_method == "remove":
                # Remove outliers
                data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
                print(f"  ✓ Outliers removed from column: {column}")
            elif outlier_method == "cap":
                # Cap outliers (replace with bounds)
                data[column] = np.where(data[column] < lower_bound, lower_bound, data[column])
                data[column] = np.where(data[column] > upper_bound, upper_bound, data[column])
                print(f"  ✓ Outliers capped in column: {column}")

if outlier_method == "cap":
    print(f"\n✓ Outliers handled using '{outlier_method}' method")
elif outlier_method == "remove" and 'outlier_count' in locals():
    print(f"\n✓ Outliers removed. New dataset shape: {data.shape}")

# Step 7: Standardize column names
print("\n7. STANDARDIZING COLUMN NAMES")
print("=" * 60)

print(f"Original column names: {list(data.columns)}")

data.columns = data.columns.str.strip()  # Remove spaces
data.columns = data.columns.str.lower()  # Convert to lowercase
data.columns = data.columns.str.replace(' ', '_')  # Replace spaces with underscores

print(f"Standardized column names: {list(data.columns)}")
print("✓ Column names standardized!")

# Step 8: Final data quality check
print("\n8. FINAL DATA QUALITY CHECK")
print("=" * 60)

print(f"\nFinal dataset shape: {data.shape}")
print(f"Rows: {data.shape[0]}")
print(f"Columns: {data.shape[1]}")
print(f"Missing values after cleaning: {data.isnull().sum().sum()}")
print(f"Duplicate rows after cleaning: {data.duplicated().sum()}")

# Step 9: Save cleaned dataset
print("\n9. SAVING CLEANED DATASET")
print("=" * 60)

output_file = 'boston_housing_cleaned.csv'
data.to_csv(output_file, index=False)
print(f"✓ Cleaned dataset saved as: {output_file}")

# Step 10: Preview cleaned data
print("\n10. PREVIEW OF CLEANED DATA")
print("=" * 60)

print("\nFirst 5 rows:")
print(data.head())

print("\nStatistical summary:")
print(data.describe())

# This is my final summary
print("\n" + "=" * 60)
print("TASK 1 COMPLETED SUCCESSFULLY!")
print("=" * 60)

print("\nSUMMARY:")
print("-" * 40)
print(f"✓ Dataset loaded and cleaned")
print(f"✓ Final shape: {data.shape[0]} rows, {data.shape[1]} columns")
print(f"✓ Missing values: {data.isnull().sum().sum()}")
print(f"✓ Duplicates removed: {duplicate_count}")
print(f"✓ Outliers handled using '{outlier_method}' method")
print(f"✓ Column names standardized")

print("\nOUTPUT FILE:")
print(f"  • {output_file} - Cleaned dataset ready for Task 2")

print("\n" + "=" * 60)
print("READY FOR TASK 2: REGRESSION ANALYSIS")
print("=" * 60)
