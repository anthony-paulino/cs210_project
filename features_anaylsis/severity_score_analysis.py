"""
The severity_score is a critical feature representing the weighted impact of collisions. It aggregates fatalities and injuries into a single numerical value.

Script Purpose: To classify collisions into severity categories that can be modeled effectively.
---------------
1. Analyze the distribution of `severity_score` in the dataset.
2. Use descriptive statistics and visualizations to understand its spread and skewness.
3. Detect patterns and imbalances in the data.
4. Determine thresholds for categorizing `severity_score` into severity classes (Low, Medium, High).

Key Outputs:
------------
- Descriptive statistics (e.g., mean, standard deviation).
- Percentile analysis to guide threshold selection.
- Visualizations (histogram and boxplot) to assess distribution.
"""

# --- Observations and Results ---
"""
Summary of Findings:
1. Descriptive Statistics:
   - Mean Severity Score: {severity_stats['mean']}
   - Standard Deviation: {severity_stats['std']}
   - Max Severity Score: {severity_stats['max']}
   - Skewed Distribution: Most scores clustered near the lower end.

2. Percentile Analysis:
   - 25th Percentile: {percentiles[0.25]} (indicates Low severity)
   - 50th Percentile (Median): {percentiles[0.50]}
   - 75th Percentile: {percentiles[0.75]} (indicates transition to Medium severity)
   - 90th Percentile: {percentiles[0.90]} (indicates transition to High severity)

Actions Taken:
---------------
- Based on this analysis, thresholds were set as:
  - **Low Severity**: 0–4
  - **Medium Severity**: 5–10
  - **High Severity**: 11+
- These thresholds balance the distribution and align with natural breaks in the data.
"""

# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data_file = "data/processed_collision_data.csv"  # Path to the processed dataset
df = pd.read_csv(data_file)

# --- Descriptive Statistics for Severity Score ---
"""
Purpose:
- Obtain basic statistics to understand the spread of `severity_score`.
- Use metrics like mean, standard deviation, and percentiles to detect skewness.
"""
severity_stats = df['severity_score'].describe()
percentiles = df['severity_score'].quantile([0.25, 0.5, 0.75, 0.9, 0.95, 0.99])

# Display basic stats
print("\n--- Basic Descriptive Statistics ---")
print(severity_stats)

# Display percentile stats
print("\n--- Percentile Analysis ---")
print(percentiles)

# --- Histogram of Severity Scores ---
"""
Purpose:
- Visualize the frequency distribution of `severity_score`.
- Detect skewness and identify natural breaks in the data for threshold selection.
"""

plt.figure(figsize=(10, 6))
sns.histplot(df['severity_score'], bins=50, kde=True, color="skyblue")  # KDE: Kernel Density Estimate for smoothness
plt.title("Histogram of Severity Scores", fontsize=14)
plt.xlabel("Severity Score", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.savefig("severity_score_histogram.png")  # Save the plot for reports
plt.show()

# --- Boxplot of Severity Scores ---
"""
Purpose:
- Use a boxplot to detect outliers and the interquartile range (IQR).
- Helps confirm potential thresholds derived from the histogram and percentiles.
"""
plt.figure(figsize=(10, 6))
sns.boxplot(x=df['severity_score'], color="lightcoral")
plt.title("Boxplot of Severity Scores", fontsize=14)
plt.xlabel("Severity Score", fontsize=12)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.savefig("severity_score_boxplot.png")  # Save the plot for reports
plt.show()

