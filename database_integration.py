import sqlite3
import pandas as pd

"""
--- Database Integration ---
This script sets up the SQLite database for the NYC Collision project.

Key Operations:
1. Create and populate the Boroughs table.
2. Create and populate the Collisions table with processed data.
3. Add indexes for optimized querying.
4. Precompute global statistics and store them in the GlobalStatistics table.
"""

# Define the database file
db_file = "data/collision_data.db"

# Load the processed dataset
data_file = "data/processed_collision_data.csv"
df = pd.read_csv(data_file)

# Define boroughs and their lat/long ranges (bounding boxes)
borough_ranges = {
    "Manhattan": {"lat_min": 40.700, "lat_max": 40.882, "lon_min": -74.019, "lon_max": -73.907},
    "Brooklyn": {"lat_min": 40.570, "lat_max": 40.739, "lon_min": -74.041, "lon_max": -73.855},
    "Queens": {"lat_min": 40.541, "lat_max": 40.800, "lon_min": -73.962, "lon_max": -73.700},
    "Bronx": {"lat_min": 40.785, "lat_max": 40.910, "lon_min": -73.933, "lon_max": -73.765},
    "Staten Island": {"lat_min": 40.477, "lat_max": 40.648, "lon_min": -74.255, "lon_max": -74.051},
}

# Function to map lat/long to a borough
def get_borough(lat, lon):
    for borough, bounds in borough_ranges.items():
        if (
            bounds["lat_min"] <= lat <= bounds["lat_max"]
            and bounds["lon_min"] <= lon <= bounds["lon_max"]
        ):
            return borough
    return "Unknown"

# Step 1: Map each data point to a borough
print("\n--- Mapping Boroughs ---")
df["borough"] = df.apply(lambda row: get_borough(row["latitude"], row["longitude"]), axis=1)
print("Mapped boroughs for all data points.")

# Step 2: Initialize the database connection
conn = sqlite3.connect(db_file)
cursor = conn.cursor()

# Step 3: Create and populate the Boroughs table
print("\n--- Creating Boroughs Table ---")
cursor.execute("""
CREATE TABLE IF NOT EXISTS Boroughs (
    borough_name TEXT PRIMARY KEY,
    lat_min REAL,
    lat_max REAL,
    lon_min REAL,
    lon_max REAL
)
""")
borough_data = [
    (borough, bounds["lat_min"], bounds["lat_max"], bounds["lon_min"], bounds["lon_max"])
    for borough, bounds in borough_ranges.items()
]
cursor.executemany("INSERT OR REPLACE INTO Boroughs VALUES (?, ?, ?, ?, ?)", borough_data)
print(f"Inserted {len(borough_data)} rows into 'Boroughs' table.")

# Step 4: Create the Collisions table
print("\n--- Creating Collisions Table ---")
cursor.execute("""
CREATE TABLE IF NOT EXISTS Collisions (
    collision_id INTEGER PRIMARY KEY AUTOINCREMENT,
    severity_score REAL,
    severity_category TEXT,
    crash_rate REAL,
    day_of_week TEXT,
    time_of_day TEXT,
    hour_of_day INTEGER,
    month INTEGER,
    vehicle_category TEXT,
    contributing_factor_category TEXT,
    location_grid TEXT,
    collision_cluster TEXT,
    latitude REAL,
    longitude REAL,
    borough TEXT
)
""")

columns_to_insert = [
    'severity_score', 'severity_category', 'crash_rate', 'day_of_week',
    'time_of_day', 'hour_of_day', 'month', 'vehicle_category', 'contributing_factor_category',
    'location_grid', 'collision_cluster', 'latitude', 'longitude', 'borough'
]

df[columns_to_insert].to_sql('Collisions', conn, if_exists='replace', index=False)
print(f"Inserted {len(df)} rows into 'Collisions' table.")

# Step 5: Create indexes to optimize querying
print("\n--- Creating Indexes ---")
cursor.execute("CREATE INDEX IF NOT EXISTS idx_location ON Collisions (latitude, longitude)")
cursor.execute("CREATE INDEX IF NOT EXISTS idx_borough ON Collisions (borough)")
cursor.execute("CREATE INDEX IF NOT EXISTS idx_datetime ON Collisions (day_of_week, time_of_day, month)")
print("Indexes created.")

# Step 6: Precompute global statistics
print("\n--- Precomputing Global Statistics ---")

# SQL-based computations for global statistics
cursor.execute("SELECT COUNT(*) FROM Collisions")
total_collisions = cursor.fetchone()[0]

# Average crash rate
cursor.execute("SELECT AVG(crash_rate) FROM Collisions")
avg_crash_rate = round(cursor.fetchone()[0], 2)

# Most frequent borough 
cursor.execute("""
SELECT borough
FROM Collisions
GROUP BY borough
ORDER BY COUNT(*) DESC
LIMIT 1
""")
most_frequent_borough = cursor.fetchone()[0]

# peak collision time 
cursor.execute("""
SELECT time_of_day
FROM Collisions
GROUP BY time_of_day
ORDER BY COUNT(*) DESC
LIMIT 1
""")
peak_collision_time = cursor.fetchone()[0]

# Severity distribution between high, medium, and low severity levels. 
cursor.execute("""
SELECT severity_category, COUNT(*) * 100.0 / (SELECT COUNT(*) FROM Collisions)
FROM Collisions
GROUP BY severity_category
""")
severity_distribution = {row[0]: round(row[1], 1) for row in cursor.fetchall()}

# Most frequent day of week
cursor.execute("""
SELECT day_of_week
FROM Collisions
GROUP BY day_of_week
ORDER BY COUNT(*) DESC
LIMIT 1
""")
most_frequent_day = cursor.fetchone()[0]

# Most common contributing factor
cursor.execute("""
SELECT contributing_factor_category
FROM Collisions
GROUP BY contributing_factor_category
ORDER BY COUNT(*) DESC
LIMIT 1
""")
most_common_factor = cursor.fetchone()[0]

# Most common vehicle type
cursor.execute("""
SELECT vehicle_category
FROM Collisions
GROUP BY vehicle_category
ORDER BY COUNT(*) DESC
LIMIT 1
""")
most_common_vehicle = cursor.fetchone()[0]

# Create or update the GlobalStatistics table
cursor.execute("""
CREATE TABLE IF NOT EXISTS GlobalStatistics (
    total_collisions INTEGER,
    avg_crash_rate REAL,
    most_frequent_borough TEXT,
    peak_collision_time TEXT,
    severity_distribution TEXT,
    most_frequent_day TEXT,
    most_common_factor TEXT,
    most_common_vehicle TEXT
)
""")

# Insert new statistics
cursor.execute("""
INSERT INTO GlobalStatistics (
    total_collisions, avg_crash_rate, most_frequent_borough,
    peak_collision_time, severity_distribution, most_frequent_day,
    most_common_factor, most_common_vehicle
)
VALUES (?, ?, ?, ?, ?, ?, ?, ?)
""", (
    total_collisions,
    avg_crash_rate,
    most_frequent_borough,
    peak_collision_time,
    str(severity_distribution),  # Serialize dictionary as string
    most_frequent_day,
    most_common_factor,
    most_common_vehicle
))
print("Global statistics table created.")
# Step 7: Commit and close the database connection
conn.commit()
conn.close()
print("\nDatabase setup complete. Connection closed.")
