Step 1: Import libraries and load data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Column names for Boston housing dataset
column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 
                'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

# Load the data (adjust filename if needed)
data = pd.read_csv('Boston house data set cleaned.csv', delim_whitespace=True, names=column_names)

# Check the data loaded correctly
print("Data loaded successfully!")
print(f"Shape: {data.shape[0]} rows, {data.shape[1]} columns")
print("\nFirst 5 rows:")
print(data.head())
print("\nColumn names:")
print(data.columns.tolist())