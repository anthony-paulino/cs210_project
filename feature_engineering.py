import pandas as pd
from sklearn.cluster import DBSCAN
import numpy as np
from sklearn.utils import resample
from download_files import check_and_download_file

''' 
-- Feature Engineering --
-------------------------
Feature Selection: 
Transform clean data to include meaningful features to enhance model performance and interpretability.
The selected features were chosen to capture the most critical aspects influencing motor vehicle collisions.

- **Severity Score**: Provides a numerical measure of collision severity by weighting injuries and fatalities.

- **Day of Week, Hour of Day, Time of Day, and Month**: Introduced temporal patterns to better identify when collisions are more likely to occur.

- **Vehicle Type Category**: Groups vehicle types into manageable categories (e.g., Passenger Vehicle, Heavy Vehicle). Different vehicle categories may have distinct risks and collision patterns.

- **Contributing Factor Category**: Maps the contributing factor to broader categories like Human Error, Environmental Conditions, and Mechanical Failure.

- **Location Grid and Collision Cluster**: Captures spatial patterns in the data, enabling to identify high-risk zones and collision hotspots.

Class Imbalance:
Class imbalance was addressed to ensure equitable model performance across all severity categories and to mitigate biases caused by the overrepresentation of less severe collisions.

- **Severity Score Balancing**: Severity categories (Low, Medium, High) were derived from severity scores, providing a manageable range of classes. To balance the dataset:
  - **Downsampling**: The majority class (`Low`) was downsampled to match the size of the minority classes. This prevents the model from being biased towards predicting the majority class.
  - **SMOTE (Synthetic Minority Oversampling Technique)**: Applied in later modeling steps to synthetically generate data for minority classes, further improving balance.

Collision Clustering:
- **Why:** Identifying spatial patterns and collision hotspots is critical for understanding risk zones. However, processing the entire dataset for clustering consumed excessive memory.
- **How:** Collision clustering was performed in chunks using the DBSCAN algorithm. Chunking allowed efficient clustering without memory overflow, as each chunk was processed independently before being recombined.
'''

# Ensure all required files are available
check_and_download_file(filepath="data/clean_collision_data.csv", url="https://drive.google.com/uc?id=12YVvDoTXSMhq65jYXiDpLW0pvLY-J8M1")

# Proceed with the rest of your script
print("All required files are available. Continuing with the script...")

# Load the cleaned dataset
df = pd.read_csv('data/clean_collision_data.csv')

# 1. Map Contributing Factors
print("\n--- Mapping Contributing Factors ---")

# Define the mapping
factor_mapping = {
    'Driver Inattention/Distraction': 'Human Error',
    'Driver Inexperience': 'Human Error',
    'Following Too Closely': 'Human Error',
    'Unsafe Speed': 'Human Error',
    'Failure to Yield Right-of-Way': 'Human Error',
    'Aggressive Driving/Road Rage': 'Human Error',
    'Backing Unsafely': 'Human Error',
    'Fell Asleep': 'Human Error',
    'Fatigued/Drowsy': 'Human Error',
    'Lost Consciousness': 'Human Error',
    'Alcohol Involvement': 'Human Error',
    'Drugs (Illegal)': 'Human Error',
    'Prescription Medication': 'Human Error',
    'Pavement Slippery': 'Environmental Factors',
    'Obstruction/Debris': 'Environmental Factors',
    'View Obstructed/Limited': 'Environmental Factors',
    'Glare': 'Environmental Factors',
    'Outside Car Distraction': 'Environmental Factors',
    'Animals Action': 'Environmental Factors',
    'Pavement Defective': 'Environmental Factors',
    'Brakes Defective': 'Mechanical/Vehicle Issues',
    'Steering Failure': 'Mechanical/Vehicle Issues',
    'Tire Failure/Inadequate': 'Mechanical/Vehicle Issues',
    'Accelerator Defective': 'Mechanical/Vehicle Issues',
    'Headlights Defective': 'Mechanical/Vehicle Issues',
    'Oversized Vehicle': 'Mechanical/Vehicle Issues',
    'Other Lighting Defects': 'Mechanical/Vehicle Issues',
    'Windshield Inadequate': 'Mechanical/Vehicle Issues',
    'Tow Hitch Defective': 'Mechanical/Vehicle Issues',
    'Vehicle Vandalism': 'Mechanical/Vehicle Issues',
    'Physical Disability': 'Mechanical/Vehicle Issues',
    'Traffic Control Disregarded': 'Traffic/Regulatory Issues',
    'Traffic Control Device Improper/Non-Working': 'Traffic/Regulatory Issues',
    'Lane Marking Improper/Inadequate': 'Traffic/Regulatory Issues',
    'Pedestrian/Bicyclist/Other Pedestrian Error/Confusion': 'Pedestrian/Cyclist Issues',
    'Unspecified': 'Unknown/Other',
    'Unknown': 'Unknown/Other',
    'Other Vehicular': 'Unknown/Other',
    'Other Electronic Device': 'Unknown/Other',
    'Cell Phone (hands-free)': 'Distracted/Occupied',
    'Cell Phone (hand-held)': 'Distracted/Occupied',
    'Using On Board Navigation Device': 'Distracted/Occupied',
    'Eating or Drinking': 'Distracted/Occupied',
    'Listening/Using Headphones': 'Distracted/Occupied',
    'Texting': 'Distracted/Occupied'
}

df['contributing_factor_category'] = df['contributing_factor_vehicle_1'].map(factor_mapping)
print("Mapped Contributing Factors to Categories.")

# 2. Severity Score and Aggregation
print("\n--- Grouping Deaths and Injuries ---")
df['total_deaths'] = df[['number_of_persons_killed', 'number_of_pedestrians_killed',
                         'number_of_cyclist_killed', 'number_of_motorist_killed']].sum(axis=1)
df['total_injuries'] = df[['number_of_persons_injured', 'number_of_pedestrians_injured',
                           'number_of_cyclist_injured', 'number_of_motorist_injured']].sum(axis=1)

# Calculate severity score
df['severity_score'] = df['total_deaths'] * 5 + df['total_injuries']
print("Calculated Severity Score.")

# Aggregate severity into categories
def aggregate_severity(score):
    if score <= 4:
        return "Low"
    elif score <= 10:
        return "Medium"
    else:
        return "High"

df['severity_category'] = df['severity_score'].apply(aggregate_severity)
print(df['severity_category'].value_counts())

# 3. Time-Based Features
print("\n--- Creating Time-Based Features ---")
df['day_of_week'] = pd.to_datetime(df['crash_datetime']).dt.day_name()
df['hour_of_day'] = pd.to_datetime(df['crash_datetime']).dt.hour
df['time_of_day'] = pd.cut(df['hour_of_day'], bins=[0, 6, 12, 18, 24],
                           labels=['Night', 'Morning', 'Afternoon', 'Evening'], right=False)
df['month'] = pd.to_datetime(df['crash_datetime']).dt.month

# 4. Vehicle Type Grouping

# Convert to lowercase
df['vehicle_type_code_1'] = df['vehicle_type_code_1'].str.lower()

# Define the mapping
vehicle_mapping = {
    'sedan': 'Passenger Vehicle',
    'station wagon/sport utility vehicle': 'Passenger Vehicle',
    'passenger vehicle': 'Passenger Vehicle',
    'sport utility / station wagon': 'Passenger Vehicle',
    'taxi': 'Taxi',
    'pick-up truck': 'Truck',
    '4 dr sedan': 'Passenger Vehicle',
    'box truck': 'Truck',
    'van': 'Van',
    'bus': 'Bus',
    'bike': 'Bike'
}

# Apply the mapping
df['vehicle_category'] = df['vehicle_type_code_1'].map(vehicle_mapping)

# Fill missing values with 'Unknown'
df.fillna({'vehicle_category': 'Unknown'}, inplace=True)

# 5. Location-Based Features
print("\n--- Deriving Location-Based Features ---")
df['lat_rounded'] = df['latitude'].round(3)
df['lon_rounded'] = df['longitude'].round(3)
df['location_grid'] = df['lat_rounded'].astype(str) + ", " + df['lon_rounded'].astype(str)

# 6. Crash Rate Feature
df['crash_rate'] = df.groupby('location_grid')['severity_score'].transform('mean')

# 7. Collision Clusters
print("\n--- Creating Collision Clusters Using Chunking ---")
def cluster_in_chunks(dataframe, chunk_size, eps=0.01, min_samples=10):
    total_clusters = []
    for start in range(0, dataframe.shape[0], chunk_size):
        end = start + chunk_size
        chunk = dataframe.iloc[start:end]
        coordinates = chunk[['latitude', 'longitude']].values
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='haversine').fit(np.radians(coordinates))
        total_clusters.extend(dbscan.labels_)
    return total_clusters

df['collision_cluster'] = cluster_in_chunks(df, chunk_size=10000)

# 8. Address Class Imbalance
print("\n--- Addressing Class Imbalance ---")
# Downsample majority class in severity category
majority_class = df[df['severity_category'] == 'Low']
minority_classes = df[df['severity_category'] != 'Low']
majority_downsampled = resample(majority_class, replace=False, n_samples=len(minority_classes), random_state=42)
df_balanced = pd.concat([majority_downsampled, minority_classes])

# 9. Save Processed Dataset
print("\n--- Saving Processed Dataset ---")
df_balanced.to_csv('data/processed_collision_data.csv', index=False)
print("Processed dataset saved.")