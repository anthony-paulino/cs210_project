"""
Analysis Script: Understanding Class Imbalance and Feature Distributions
-------------------------------------------------------------------------
This script is part of the data processing phase and serves the following purposes:
1. Analyze the distribution of key features (`severity_score`, `severity_category`, and `collision_cluster`) to identify potential class imbalances.
2. Support data preprocessing decisions, such as thresholding or grouping, to address imbalances effectively.
3. Generate visualizations and summary statistics to provide insights into the dataset and guide adjustments to feature engineering.

Inputs:
- `data/processed_collision_data.csv`: A processed dataset that includes key features for analysis.

Outputs:
- Insights into the distribution of severity scores, severity categories, and collision clusters.
- Visualizations (e.g., bar plots, histograms) to identify patterns and imbalances.
- Data-driven justifications for decisions made during processing and modeling.

Key Adjustments and Results:
----------------------------
1. **Severity Score**:
   - Represents the weighted impact of a collision based on injuries and fatalities.
   - Analysis showed a significant skew toward low severity scores.
   - Action Taken:
     - Severity scores were aggregated into three categories: Low, Medium, and High (stored in `severity_category`).
     - These thresholds were determined based on percentile analysis and visualizations.
     - SMOTE was later applied to balance these severity categories during model training.

2. **Collision Cluster**:
   - Groups collisions based on spatial proximity (e.g., DBSCAN).
   - Analysis revealed dominance of certain clusters and sparsity in others, including noise points (-1).
   - Action Taken:
     - Noise points and sparse clusters were reviewed during modeling.
"""

# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt

# Load the processed dataset
df = pd.read_csv('data/processed_collision_data.csv')

# --- Severity Score Analysis ---
"""
Severity Score:
- Represents the numerical severity of collisions, derived from fatalities and injuries.
- Objective: Analyze its distribution to detect potential imbalances in severity levels.
"""
print("\n--- Severity Score Analysis ---")
severity_counts = df['severity_score'].value_counts()  # Count occurrences of each severity score
severity_proportions = severity_counts / severity_counts.sum() * 100  # Calculate proportions (percentage)

# Print findings to console
print("Counts for each severity score:")
print(severity_counts)
print("\nProportions for each severity score (%):")
print(severity_proportions)

# Visualization: Severity Score Distribution
plt.figure(figsize=(10, 6))
severity_counts.plot(kind='bar', color='skyblue', alpha=0.8)  # Bar plot for severity score counts
plt.title("Severity Score Distribution", fontsize=14)
plt.xlabel("Severity Score", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)  # Add gridlines for readability
plt.show()

# --- Severity Category Analysis ---
"""
Severity Category:
- Derived from severity scores using defined thresholds: Low, Medium, High.
- Objective: Examine the distribution of severity categories for potential imbalances.
"""
print("\n--- Severity Category Analysis ---")
category_counts = df['severity_category'].value_counts()  # Count occurrences of each severity category
category_proportions = category_counts / category_counts.sum() * 100  # Calculate proportions (percentage)

# Print findings to console
print("Counts for each severity category:")
print(category_counts)
print("\nProportions for each severity category (%):")
print(category_proportions)

# Visualization: Severity Category Distribution
plt.figure(figsize=(10, 6))
category_counts.plot(kind='bar', color='lightgreen', alpha=0.8)  # Bar plot for severity category counts
plt.title("Severity Category Distribution", fontsize=14)
plt.xlabel("Severity Category", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)  # Add gridlines for readability
plt.show()

# --- Collision Cluster Analysis ---
"""
Collision Cluster:
- Groups collisions into clusters based on spatial proximity (e.g., DBSCAN).
- Objective: Understand the distribution of clusters to detect dominance or sparsity.
"""
print("\n--- Collision Cluster Analysis ---")
cluster_counts = df['collision_cluster'].value_counts()  # Count occurrences of each collision cluster
cluster_proportions = cluster_counts / cluster_counts.sum() * 100  # Calculate proportions (percentage)

# Print findings to console
print("Counts for each collision cluster:")
print(cluster_counts)
print("\nProportions for each collision cluster (%):")
print(cluster_proportions)

# Visualization: Collision Cluster Distribution
plt.figure(figsize=(10, 6))
cluster_counts.plot(kind='bar', color='lightcoral', alpha=0.8)  # Bar plot for collision cluster counts
plt.title("Collision Cluster Distribution", fontsize=14)
plt.xlabel("Collision Cluster", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)  # Add gridlines for readability
plt.show()
