import pandas as pd
from download_files import check_and_download_file 

"""
--- Exploratory Data Analysis ---
---------------------------------
Geographic Columns:
- boroughs, zip_code, on_street_name, cross_street_name, off_street_name is supplementary and not critical for the modeling or visualization.
A lot of records have the following columns missing, Reverse geocoding (to derive these columns through the longitude and latitude) is time-consuming and API-dependent, especially with large datasets.
We attempted to impute these values through reverse geocoding but it was taking sn extremely long time. 
- A possible solution is to have a set range of latitude and longitude to represent boroughs and other geo details.

Contributing Factors:
- For columns that contained "Unspecified" or "Unknown" in contributing_factor_vehicle_1
we checked contributing factors (`contributing_factor_vehicle_2` to `contributing_factor_vehicle_5`) 
to determine whether they would provide additional information other than "Unspecified" or "Unknown".
   the majority of values were found to be either "Unknown" or "Unspecified" as well.
- Decision: Retain only `contributing_factor_vehicle_1`and drop the secondary factor columns.

Vehicle Types:
- A similar analysis was conducted for the column vehicle_type_code_1
that contained "Unspecified" or "Unknown", we analyzed the values for `vehicle_type_code_2` to `vehicle_type_code_5` for those records.
They did not contain anything useful, and contained "Unspecified" or "Unknown" as well.
- Decision: Retain only `vehicle_type_code_1` for analysis and modeling and drop the secondary vehicle type columns.

"""

# Ensure all required files are available
check_and_download_file(filepath="data/clean_collision_data.csv", url="https://drive.google.com/uc?id=12YVvDoTXSMhq65jYXiDpLW0pvLY-J8M1")

# Proceed with the rest of your script
print("All required files are available. Continuing with the script...")

df = pd.read_csv('data/clean_collision_data.csv')

# Analyze unique values in contributing factors and vehicle types
print("\n--- Possible Values Analysis ---")
print("Unique values in 'contributing_factor_vehicle_1':")
print(df['contributing_factor_vehicle_1'].value_counts())
print("Unique values in 'contributing_factor_vehicle_2':")
print(df['contributing_factor_vehicle_2'].value_counts())
print("Unique values in 'contributing_factor_vehicle_3':")
print(df['contributing_factor_vehicle_3'].value_counts())
print("Unique values in 'contributing_factor_vehicle_4':")
print(df['contributing_factor_vehicle_4'].value_counts())
print("Unique values in 'contributing_factor_vehicle_5':")
print(df['contributing_factor_vehicle_5'].value_counts())

print("\nUnique values in 'vehicle_type_code_1':")
print(df['vehicle_type_code_1'].value_counts()[:15])
print("\nUnique values in 'vehicle_type_code_2':")
print(df['vehicle_type_code_2'].value_counts())
print("\nUnique values in 'vehicle_type_code_3':")
print(df['vehicle_type_code_3'].value_counts())
print("\nUnique values in 'vehicle_type_code_4':")
print(df['vehicle_type_code_4'].value_counts())
print("\nUnique values in 'vehicle_type_code_5':")
print(df['vehicle_type_code_5'].value_counts())

# Identify records where type_1 is "Unknown" or "Unspecified"
unknown_vehicle_type1 = df[
    df['vehicle_type_code_1'].isin(['Unknown', 'Unspecified'])
]

print("\nCounts for 'Unknown' and 'Unspecified' in vehicle_type_code_1:")
print(unknown_vehicle_type1['vehicle_type_code_1'].value_counts())

# Check the other vehicle types for these records to see if they provide additional information
for i in range(2, 6):  # Check vehicle_type_code_2 to vehicle_type_code_5
    type_col = f'vehicle_type_code_{i}'
    if type_col in df.columns:
        print(f"\nUnique values in {type_col} for records where type_1 is 'Unknown' or 'Unspecified':")
        print(unknown_vehicle_type1[type_col].value_counts())

# Identify records where contributing_factor_vehicle_1 is "Unknown" or "Unspecified"
unknown_contributing_factors_1 = df[
    df['contributing_factor_vehicle_1'].isin(['Unknown', 'Unspecified'])
]

print("\nCounts for 'Unknown' and 'Unspecified' in contributing_factor_vehicle_1:")
print(unknown_contributing_factors_1['contributing_factor_vehicle_1'].value_counts())

# Check secondary contributing factors for these records
for i in range(2, 6):  # Check contributing_factor_vehicle_2 to contributing_factor_vehicle_5
    factor_col = f'contributing_factor_vehicle_{i}'
    if factor_col in df.columns:
        print(f"\nUnique values in {factor_col} for records where contributing_factor_vehicle_1 is 'Unknown' or 'Unspecified':")
        print(unknown_contributing_factors_1[factor_col].value_counts())


# Function to determine the primary contributing factor using fallback
def determine_primary_contributing_factor(row):
    if row['contributing_factor_vehicle_1'] not in ["Unknown", "Unspecified"]:
        return row['contributing_factor_vehicle_1']
    # Check secondary columns for non-"Unknown"/"Unspecified" values
    for col in ['contributing_factor_vehicle_2', 'contributing_factor_vehicle_3', 
                'contributing_factor_vehicle_4', 'contributing_factor_vehicle_5']:
        if row[col] not in ["Unknown", "Unspecified"]:
            return row[col]
    return row['contributing_factor_vehicle_1']  # Default to type_1 if no fallback found

# Apply the fallback mechanism
df['primary_contributing_factor'] = df.apply(determine_primary_contributing_factor, axis=1)

# Drop secondary factor columns as they are no longer needed
df = df.drop(columns=[
    'contributing_factor_vehicle_2', 'contributing_factor_vehicle_3',
    'contributing_factor_vehicle_4', 'contributing_factor_vehicle_5'
], errors='ignore')

# Verify the counts for 'Unknown' and 'Unspecified' in the finalized column
print("\nFinal 'primary_contributing_factor' Column Value Counts:")
print(df['primary_contributing_factor'].value_counts())

# Check how the counts for 'Unknown' and 'Unspecified' dropped
unknown_count = df['primary_contributing_factor'].value_counts().get('Unknown', 0)
unspecified_count = df['primary_contributing_factor'].value_counts().get('Unspecified', 0)
print(f"\nCounts after fallback - 'Unknown': {unknown_count}, 'Unspecified': {unspecified_count}")
