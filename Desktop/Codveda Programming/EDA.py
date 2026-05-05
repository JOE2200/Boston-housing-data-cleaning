# Task 2: Exploratory Data Analysis (EDA)
# Boston Housing Dataset

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("=" * 60)
print("TASK 2: EXPLORATORY DATA ANALYSIS (EDA)")
print("Boston Housing Dataset")
print("=" * 60)

# ── 0. Setup ────────────────────────────────────────────────
os.makedirs("eda_plots", exist_ok=True)
plt.rcParams.update({'figure.dpi': 150, 'figure.figsize': (10, 6)})

# ── 1. Load Dataset ─────────────────────────────────────────
print("\n1. LOADING THE DATASET")
print("-" * 60)

column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
                'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

file_name = "Data Set For Task/4) house Prediction Data Set.csv"
data = pd.read_csv(file_name, sep=r'\s+', names=column_names, engine='python')

print(f"✓ Loaded: {file_name}")
print(f"  Shape: {data.shape[0]} rows × {data.shape[1]} columns")

# ── 2. Summary Statistics ────────────────────────────────────
print("\n2. SUMMARY STATISTICS")
print("=" * 60)

stats = data.describe()
print(stats.round(2).to_string())

print("\nAdditional Statistics:")
print("-" * 40)
for col in data.columns:
    mode_val = data[col].mode()[0]
    print(f"  {col:<8}  mean={data[col].mean():.2f}  median={data[col].median():.2f}  "
          f"mode={mode_val:.2f}  std={data[col].std():.2f}")

# ── 3. Data Types & Missing Values ──────────────────────────
print("\n3. DATA TYPES & MISSING VALUES")
print("=" * 60)
print(data.dtypes)
print(f"\nMissing values: {data.isnull().sum().sum()}")

# ── 4. Histograms ────────────────────────────────────────────
print("\n4. PLOTTING HISTOGRAMS (Distribution of all features)")
print("=" * 60)

fig, axes = plt.subplots(3, 5, figsize=(18, 12))
axes = axes.flatten()
colors = sns.color_palette("husl", len(column_names))

for i, col in enumerate(column_names):
    axes[i].hist(data[col], bins=30, color=colors[i], edgecolor='black', alpha=0.85)
    axes[i].set_title(col, fontsize=12, fontweight='bold')
    axes[i].set_xlabel('Value')
    axes[i].set_ylabel('Frequency')
    axes[i].grid(axis='y', linestyle='--', alpha=0.5)

# Hide the last unused subplot
axes[-1].set_visible(False)

fig.suptitle('Boston Housing Dataset – Feature Distributions', fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('eda_plots/01_histograms.png', bbox_inches='tight')
plt.close()
print("  ✓ Saved: eda_plots/01_histograms.png")

# ── 5. Boxplots ───────────────────────────────────────────────
print("\n5. PLOTTING BOXPLOTS (Outlier detection)")
print("=" * 60)

fig, axes = plt.subplots(3, 5, figsize=(18, 12))
axes = axes.flatten()

for i, col in enumerate(column_names):
    axes[i].boxplot(data[col], patch_artist=True,
                    boxprops=dict(facecolor=colors[i], alpha=0.7),
                    medianprops=dict(color='black', linewidth=2))
    axes[i].set_title(col, fontsize=12, fontweight='bold')
    axes[i].set_ylabel('Value')
    axes[i].grid(axis='y', linestyle='--', alpha=0.5)

axes[-1].set_visible(False)
fig.suptitle('Boston Housing Dataset – Boxplots (Outlier Detection)', fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('eda_plots/02_boxplots.png', bbox_inches='tight')
plt.close()
print("  ✓ Saved: eda_plots/02_boxplots.png")

# ── 6. Scatter Plots (vs MEDV) ────────────────────────────────
print("\n6. SCATTER PLOTS (Features vs House Price MEDV)")
print("=" * 60)

features_of_interest = ['RM', 'LSTAT', 'PTRATIO', 'CRIM', 'NOX', 'DIS',
                         'TAX', 'AGE', 'INDUS', 'RAD', 'B', 'ZN']

fig, axes = plt.subplots(3, 4, figsize=(18, 14))
axes = axes.flatten()

for i, col in enumerate(features_of_interest):
    axes[i].scatter(data[col], data['MEDV'], alpha=0.5, color=colors[i], edgecolors='none', s=20)
    
    # Add a simple linear trend line
    z = np.polyfit(data[col], data['MEDV'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(data[col].min(), data[col].max(), 100)
    axes[i].plot(x_line, p(x_line), "r--", linewidth=1.5, label='Trend')
    
    axes[i].set_xlabel(col, fontsize=10)
    axes[i].set_ylabel('MEDV (House Price)', fontsize=10)
    axes[i].set_title(f'{col} vs MEDV', fontsize=11, fontweight='bold')
    axes[i].legend(fontsize=8)
    axes[i].grid(linestyle='--', alpha=0.4)

fig.suptitle('Boston Housing – Feature vs House Price (MEDV) Scatter Plots',
             fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('eda_plots/03_scatter_plots.png', bbox_inches='tight')
plt.close()
print("  ✓ Saved: eda_plots/03_scatter_plots.png")

# ── 7. Correlation Matrix Heatmap ────────────────────────────
print("\n7. CORRELATION ANALYSIS")
print("=" * 60)

corr_matrix = data.corr()

print("\nCorrelation with MEDV (House Price):")
print("-" * 40)
medv_corr = corr_matrix['MEDV'].drop('MEDV').sort_values(ascending=False)
for col, val in medv_corr.items():
    bar = '█' * int(abs(val) * 20)
    direction = '+' if val > 0 else '-'
    print(f"  {col:<8} {direction}{bar} {val:.3f}")

fig, ax = plt.subplots(figsize=(14, 11))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdYlGn',
            center=0, vmin=-1, vmax=1, square=True, linewidths=0.5,
            annot_kws={'size': 9}, ax=ax,
            cbar_kws={'shrink': 0.8, 'label': 'Correlation Coefficient'})
ax.set_title('Boston Housing Dataset – Correlation Matrix Heatmap',
             fontsize=15, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('eda_plots/04_correlation_heatmap.png', bbox_inches='tight')
plt.close()
print("\n  ✓ Saved: eda_plots/04_correlation_heatmap.png")

# ── 8. MEDV Distribution ─────────────────────────────────────
print("\n8. TARGET VARIABLE (MEDV) DISTRIBUTION")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram with KDE
axes[0].hist(data['MEDV'], bins=30, color='steelblue', edgecolor='black', alpha=0.75, density=True)
data['MEDV'].plot.kde(ax=axes[0], color='darkred', linewidth=2)
axes[0].axvline(data['MEDV'].mean(),   color='orange', linestyle='--', linewidth=2, label=f"Mean: {data['MEDV'].mean():.2f}")
axes[0].axvline(data['MEDV'].median(), color='green',  linestyle='--', linewidth=2, label=f"Median: {data['MEDV'].median():.2f}")
axes[0].set_title('MEDV Distribution (Histogram + KDE)', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Median House Value ($1000s)')
axes[0].set_ylabel('Density')
axes[0].legend()
axes[0].grid(linestyle='--', alpha=0.5)

# Boxplot
axes[1].boxplot(data['MEDV'], patch_artist=True,
                boxprops=dict(facecolor='steelblue', alpha=0.7),
                medianprops=dict(color='darkred', linewidth=2.5),
                flierprops=dict(marker='o', markerfacecolor='red', markersize=5))
axes[1].set_title('MEDV Boxplot', fontsize=13, fontweight='bold')
axes[1].set_ylabel('Median House Value ($1000s)')
axes[1].grid(axis='y', linestyle='--', alpha=0.5)

fig.suptitle('Target Variable – MEDV (Median House Value)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('eda_plots/05_medv_distribution.png', bbox_inches='tight')
plt.close()
print("  ✓ Saved: eda_plots/05_medv_distribution.png")

# ── 9. Top Correlations Pairplot ─────────────────────────────
print("\n9. PAIRPLOT (Top 4 correlated features with MEDV)")
print("=" * 60)

top_features = medv_corr.abs().nlargest(4).index.tolist() + ['MEDV']
pair_data = data[top_features]

fig = plt.figure(figsize=(14, 12))
pd.plotting.scatter_matrix(pair_data, figsize=(14, 12), alpha=0.4,
                            diagonal='hist', color='steelblue',
                            hist_kwds={'bins': 25, 'edgecolor': 'black'})
plt.suptitle('Pairplot – Top 4 Features Most Correlated with MEDV',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('eda_plots/06_pairplot_top_features.png', bbox_inches='tight')
plt.close()
print("  ✓ Saved: eda_plots/06_pairplot_top_features.png")

# ── 10. Final Summary ─────────────────────────────────────────
print("\n" + "=" * 60)
print("TASK 2 COMPLETED SUCCESSFULLY!")
print("=" * 60)

print("\nKEY INSIGHTS:")
print("-" * 40)
print(f"  • Dataset has {data.shape[0]} records and {data.shape[1]} features")
print(f"  • Target variable MEDV: mean=${data['MEDV'].mean():.2f}k, median=${data['MEDV'].median():.2f}k")
print(f"  • Strongest POSITIVE correlation with MEDV: RM ({corr_matrix.loc['RM','MEDV']:.3f})")
print(f"  • Strongest NEGATIVE correlation with MEDV: LSTAT ({corr_matrix.loc['LSTAT','MEDV']:.3f})")
print(f"  • CHAS (river boundary) has low correlation ({corr_matrix.loc['CHAS','MEDV']:.3f})")

print("\nOUTPUT PLOTS (saved in eda_plots/):")
print("  01_histograms.png          – Distribution of all features")
print("  02_boxplots.png            – Outlier detection per feature")
print("  03_scatter_plots.png       – Each feature vs MEDV with trend line")
print("  04_correlation_heatmap.png – Full correlation matrix")
print("  05_medv_distribution.png   – Target variable analysis")
print("  06_pairplot_top_features.png – Top features pairplot")

print("\n" + "=" * 60)
print("READY FOR TASK 3: BASIC DATA VISUALIZATION")
print("=" * 60)
