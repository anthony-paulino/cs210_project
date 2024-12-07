"""
Script Purpose: Wanted to assign appropriate thresholds for the categorization of severity score into severity levels.
---------------
1. Refine the categorization of `severity_score` into severity levels (Low, Medium, High).
2. Assign severity levels based on thresholds derived from data analysis.
3. Generate summary statistics (mean, min, max, count) for each severity level.
4. Save the updated statistics to a CSV file for review and documentation.

Key Outputs:
------------
- Summary statistics for each severity level saved as `data/severity_level_stats_updated.csv`.
"""

# --- Summary of Results ---
"""
Key Findings:
1. Severity Levels:
   - Low: Represents collisions with minor severity (most frequent category).
   - Medium: Represents collisions with moderate severity.
   - High: Represents collisions with severe impacts.

2. Summary Statistics:
   - Mean: Helps understand the average severity score for each level.
   - Count: Highlights the distribution of records across severity levels.
   - Min/Max: Confirms threshold boundaries are applied correctly.

Impact:
- These levels provide meaningful groupings for model training.
- Results are consistent with prior analysis, ensuring validity of thresholds.
"""

# Import necessary libraries
import pandas as pd

# Load the processed dataset
file_path = "data/processed_collision_data.csv"  # Path to the processed dataset
df = pd.read_csv(file_path)

# --- Define Severity Thresholds ---
"""
Purpose:
- Use thresholds derived from the analysis of `severity_score`.
- Categorize severity into three levels for modeling and reporting.

Thresholds:
- Low: Scores <= 4.0
- Medium: Scores between 4.1 and 10.0
- High: Scores > 10.0
"""
low_threshold = 4.0
medium_threshold = 10.0

# --- Assign Severity Levels ---
"""
Purpose:
- Apply thresholds to assign a severity level to each record in the dataset.
- Adds a new column `severity_level` to the dataset.
"""
def assign_severity_level(score):
    if score <= low_threshold:
        return "Low"
    elif low_threshold < score <= medium_threshold:
        return "Medium"
    else:
        return "High"

df['severity_level'] = df['severity_score'].apply(assign_severity_level)

# --- Group and Calculate Statistics ---
"""
Purpose:
- Group data by `severity_level` and calculate summary statistics:
  - Mean: Average severity score for each level.
  - Min: Minimum severity score in each level.
  - Max: Maximum severity score in each level.
  - Count: Total number of records in each level.
"""
severity_stats = df.groupby('severity_level')['severity_score'].agg(['mean', 'min', 'max', 'count'])

# Save statistics to a CSV file
output_path = "data/severity_level_stats_updated.csv"
severity_stats.to_csv(output_path)
print(f"Severity statistics saved to {output_path}.")

# --- Display Statistics ---
"""
Purpose:
- Print updated statistics for quick review and validation.
"""
print("\n--- Updated Severity Level Statistics ---")
print(severity_stats)

