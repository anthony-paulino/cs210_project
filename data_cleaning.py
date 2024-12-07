# Import necessary libraries
import pandas as pd
from download_files import check_and_download_file  

''' 
--- Data Ingestion and Cleaning ---
-----------------------------------
Purpose:
1. Load and inspect raw motor vehicle collision data.
2. Perform data cleaning to prepare the dataset for further analysis and modeling.

Key Cleaning Steps:
1. Handle duplicate records to avoid redundant data.
2. Manage missing values:
   - Drop rows missing critical geographic or injury/fatality data.
   - Fill missing values in categorical columns with "Unknown."
   - Drop supplementary columns that are not critical for modeling or visualization.
3. Standardize column names for consistency.
4. Convert data types to their appropriate formats (e.g., datetime, category).
5. Combine crash date and time into a single timestamp for easier temporal analysis.

Why Certain Columns Were Dropped:
- Columns like `borough` and `zip_code` were supplementary and costly to impute accurately. 
    - An attempt was made using reverse geocoding based on longitude and latitude to derive the borough and ZIP code of each location point but was to costly and time-intensive.
    - We implemented a solution in the database integration step to assign each location point to a borough.
- Supplementary street information (e.g., `on_street_name`) was not relevant to modeling or visualization.
- The `collision_id` column was removed as we will generate our own unique IDs.

Output:
- A cleaned dataset saved as `data/clean_collision_data.csv`.
'''

# Ensure all required files are available
check_and_download_file(filepath="data/raw_dataset.csv", url="https://drive.google.com/uc?id=15QJIa6AxucXFIwCEITNL_uioh8cEj8Er")

# Proceed with the rest of your script
print("All required files are available. Continuing with the script...")

# Step 1: Load the Data
file_path = "data/raw_dataset.csv"
df = pd.read_csv(file_path, low_memory=False)  # Load the dataset

# Step 2: Inspect the Data
print("\n--- Data Inspection ---")
print("Basic Information:")
print(df.info())  # Overview of data types and non-null counts

print("\nSummary Statistics:")
print(df.describe())  # Summary statistics for numerical columns

# Step 3: Handle Duplicates
print("\n--- Handling Duplicates ---")
duplicates = df.duplicated().sum()
print(f"Number of Duplicate Rows: {duplicates}")

if duplicates > 0:
    print("Removing duplicate rows...")
    df = df.drop_duplicates()

# Step 4: Standardize Column Names
print("\n--- Standardizing Column Names ---")
# Ensure column names are consistent and lowercase with underscores
df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
print("Column Names After Standardization:")
print(df.columns)

# Step 5: Handle Missing Values
print("\n--- Handling Missing Values ---")
missing_values = df.isnull().sum()
print("Missing Values per Column:")
print(missing_values)

# Step 5.1: Drop rows missing critical geographic data
geo_columns_critical = ['latitude', 'longitude']
print(f"Dropping rows missing critical geographic data: {geo_columns_critical}")
df = df.dropna(subset=geo_columns_critical, how='any')

# Step 5.2: Drop unnecessary columns
columns_to_drop = ['borough', 'zip_code', 'on_street_name', 'cross_street_name', 'off_street_name', 'collision_id']
print(f"Dropping unnecessary columns: {columns_to_drop}")
df = df.drop(columns=columns_to_drop, errors='ignore')

# Step 5.3: Fill missing values in contributing factors and vehicle types
print("Filling missing values in contributing factors and vehicle types with 'Unknown'...")
for col in [
    'contributing_factor_vehicle_1', 'contributing_factor_vehicle_2',
    'contributing_factor_vehicle_3', 'contributing_factor_vehicle_4',
    'contributing_factor_vehicle_5', 'vehicle_type_code_1',
    'vehicle_type_code_2', 'vehicle_type_code_3',
    'vehicle_type_code_4', 'vehicle_type_code_5'
]:
    if col in df.columns:
        df[col] = df[col].fillna('Unknown')

# Step 5.4: Drop rows missing all injury or fatality data
injury_cols = [
    'number_of_persons_injured', 'number_of_persons_killed',
    'number_of_pedestrians_injured', 'number_of_pedestrians_killed',
    'number_of_cyclist_injured', 'number_of_cyclist_killed',
    'number_of_motorist_injured', 'number_of_motorist_killed'
]
print(f"Dropping rows missing all injury/fatality data: {injury_cols}")
df = df.dropna(subset=injury_cols, how='all')

# Fill remaining missing values in injury and fatality columns with zeros
df[injury_cols] = df[injury_cols].fillna(0)

# Step 6: Data Type Conversions
print("\n--- Data Type Conversions ---")

# Combine crash_date and crash_time into a single datetime column
print("Combining 'crash_date' and 'crash_time' into a single 'crash_datetime' column...")
df['crash_date'] = pd.to_datetime(df['crash_date'], errors='coerce')  # Convert crash_date to datetime
df['crash_datetime'] = pd.to_datetime(
    df['crash_date'].astype(str) + ' ' + df['crash_time'], errors='coerce'
)
df = df.drop(columns=['crash_date', 'crash_time'], errors='ignore')  # Drop the original columns

# Convert relevant columns to 'category' type for memory efficiency and better modeling
categorical_columns = [
    'contributing_factor_vehicle_1', 'contributing_factor_vehicle_2',
    'contributing_factor_vehicle_3', 'contributing_factor_vehicle_4', 'contributing_factor_vehicle_5',
    'vehicle_type_code_1', 'vehicle_type_code_2', 'vehicle_type_code_3',
    'vehicle_type_code_4', 'vehicle_type_code_5'
]
print("Converting relevant columns to 'category' type...")
for col in categorical_columns:
    if col in df.columns:
        df[col] = df[col].astype('category')

# Step 7: Final Checks and Saving
print("\n--- Final Checks ---")

# Check for any remaining missing values
print("Missing Values After Processing:")
print(df.isna().sum())

# Confirm data types
print("\nData Types After Processing:")
print(df.dtypes)

# Print dataset size after cleaning
print(f"Number of records after cleaning: {df.shape[0]}\n")

# Save the cleaned dataset to a CSV file
output_file = "data/clean_collision_data.csv"
df.to_csv(output_file, index=False)
print(f"\nCleaned dataset saved to {output_file}")
