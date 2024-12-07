import sqlite3
import pandas as pd
import streamlit as st
import folium
from folium.plugins import HeatMap
import joblib
from download_files import check_and_download_file

# --- NYC Collision Data Application ---
# ---------------------------------------
# Purpose:
# 1. Visualize motor vehicle collision data and provide severity predictions.
# 2. Integrate SQL operations for efficient querying and filtering.
# 3. Offer real-time and user-specific statistics dynamically.

# Ensure all required files are available
check_and_download_file(filepath="models/random_forest_model.pkl", url="https://drive.google.com/uc?id=1dwqypIt_eLYZM1tM8UR7g7HBuwxy17df")

# Proceed with the rest of your script
print("All required files are available. Continuing with the script...")

# Load trained Random Forest model, preprocessor, encoder
rf_model = joblib.load("models/random_forest_model.pkl")
label_encoder = joblib.load("encoders/label_encoder.pkl")

# Get the processed feature names
with open("models/feature_names.txt", "r") as f:
    feature_names = f.read().splitlines()

# Align the user input to the prediction model input structure 
def create_aligned_input(input_data, feature_names):
    # Initialize aligned DataFrame with all zeros
    aligned_input = pd.DataFrame(0, index=[0], columns=feature_names, dtype='float64')

    # Ensure all input data for categorical features is treated as strings
    input_data = input_data.astype(str)

    # Map numerical features
    for num_feature in [f for f in feature_names if f.startswith("num__")]:
        try:
            # Map numerical features based on the suffix
            feature_name = num_feature.split("__")[1]
            aligned_input.at[0, num_feature] = float(input_data[feature_name].values[0])
        except KeyError:
            print(f"Numerical Feature Missing: {num_feature}")

    # Map categorical features
    for cat_feature in [f for f in feature_names if f.startswith("cat__")]:
        try:
            # Extract the category name from the feature
            prefix, category_name = cat_feature.split("__")

            column_name = category_name.rsplit("_", 1)[0]  # Extract column base name
            # Check if the input data has the corresponding category
            if column_name in input_data.columns and input_data[column_name].iloc[0] == cat_feature.rsplit("_")[-1]:

                aligned_input.at[0, cat_feature] = 1
        except KeyError:
            print(f"Categorical Feature Missing: {cat_feature}")

    return aligned_input

# Fetch Collision data filtered by the user input values (via SQL)
def load_filtered_data(boroughs, time_filter_type, time_value, months, days_of_week, factors, vehicles):
    """
    Query filtered collision data from the SQLite database based on user inputs.
    """
    conn = sqlite3.connect("data/collision_data.db")
    query = f"""
    SELECT 
        latitude, longitude, severity_score, severity_category, borough,
        day_of_week, time_of_day, crash_rate, contributing_factor_category,
        vehicle_category, collision_cluster, month
    FROM Collisions
    WHERE borough IN ({','.join(['?'] * len(boroughs))})
    AND month IN ({','.join(['?'] * len(months))})
    AND day_of_week IN ({','.join(['?'] * len(days_of_week))})
    AND contributing_factor_category IN ({','.join(['?'] * len(factors))})
    AND vehicle_category IN ({','.join(['?'] * len(vehicles))})
    """
    if time_filter_type == "Time of Day":
        query += " AND time_of_day IN ({})".format(",".join(["?"] * len(time_value)))
        params = boroughs + months + days_of_week + factors + vehicles + time_value
    else:
        query += " AND hour_of_day BETWEEN ? AND ?"
        params = boroughs + months + days_of_week + factors + vehicles + list(time_value)

    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df

@st.cache_data
def load_global_stats():
    """
    Fetch precomputed global statistics from the database.
    """
    conn = sqlite3.connect("data/collision_data.db")
    query = "SELECT * FROM GlobalStatistics;"
    stats = pd.read_sql_query(query, conn)
    conn.close()
    return stats.to_dict(orient="records")[0]
# Fetch unique values of a specified column in the Collision table (via SQL)
@st.cache_data
def get_unique_values(column_name):
    """
    Fetch unique values for a specific column from the Collisions table.
    """
    conn = sqlite3.connect("data/collision_data.db")
    query = f"SELECT DISTINCT {column_name} FROM Collisions ORDER BY {column_name};"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df[column_name].dropna().tolist()

# Streamlit App
st.title("NYC Collision Data & Prediction")

# Global Statistics
global_stats = load_global_stats()
st.sidebar.header("Global Collision Summary")
st.sidebar.json(global_stats)

# Sidebar Filters
st.sidebar.header("Filter Collision Data")
unique_boroughs = get_unique_values("borough")
# Stay within in NYC
unique_boroughs.remove("Unknown")
borough = st.sidebar.multiselect("Select Borough(s)", unique_boroughs, default=["Manhattan"])

month_names = [
    "January", "February", "March", "April", "May", "June", 
    "July", "August", "September", "October", "November", "December"
]
month = st.sidebar.multiselect(
    "Select Month(s)", range(1, 13), format_func=lambda x: f"{x} - {month_names[x-1]}", default=[1]
)

unique_days = get_unique_values("day_of_week")
day_of_week = st.sidebar.multiselect("Select Day(s) of the Week", unique_days, default=['Monday'])

unique_factors = get_unique_values("contributing_factor_category")
factor = st.sidebar.multiselect("Select Contributing Factor(s)", unique_factors, default=['Human Error'])

unique_vehicles = get_unique_values("vehicle_category")
vehicle = st.sidebar.multiselect("Select Vehicle Type(s)", unique_vehicles, default=['Passenger Vehicle'])

time_filter_type = st.sidebar.radio("Time Filter Type", ["Time of Day", "Time Range"])
if time_filter_type == "Time of Day":
    unique_time_of_day = get_unique_values("time_of_day")
    time_of_day = st.sidebar.multiselect("Select Time(s) of Day", unique_time_of_day, default=["Morning"])
    filtered_df = load_filtered_data(borough, time_filter_type, time_of_day, month, day_of_week, factor, vehicle)
else:
    time_range = st.sidebar.slider("Select Time Range (24-hour format)", 0, 24, (0, 24), step=1)
    filtered_df = load_filtered_data(borough, time_filter_type, time_range, month, day_of_week, factor, vehicle)

# User-Specific Statistics (Dynamic)
st.sidebar.header("User-Specific Collision Statistics")
if not filtered_df.empty:
    # Map selected months to their English names
    selected_months = [month_names[m - 1] for m in month]
    
    user_stats = {
        "Filtered Collisions": len(filtered_df),
        "Average Crash Rate": round(filtered_df["crash_rate"].mean(), 2),
        "Peak Collision Time": filtered_df["time_of_day"].mode()[0],
        "Severity Distribution (%)": filtered_df["severity_category"].value_counts(normalize=True).mul(100).round(1).to_dict(),
        "Selected Months": ", ".join(selected_months),  # Join month names as a string
        "Selected Day(s)": ", ".join(day_of_week),
        "Contributing Factor(s)": ", ".join(factor),
        "Vehicle Type(s)": ", ".join(vehicle),
    }
    st.sidebar.json(user_stats)

st.info("🌟 **Don't miss out! Scroll down to explore our collision severity prediction feature.** 🚗💥")

# Display Heatmap
st.write("### Heatmap of Collisions")
m = folium.Map(location=[40.7128, -74.0060], zoom_start=12)
heat_data = filtered_df[["latitude", "longitude"]].dropna().values.tolist()
HeatMap(heat_data).add_to(m)
st.components.v1.html(m._repr_html_(), height=600)


# Predict Collision Section
st.write("### Predict Collision Severity")

# Fetch borough ranges for latitude and longitude (via SQL)
def get_borough_ranges():
    """
    Fetch latitude and longitude ranges for each borough from the Boroughs table.
    """
    conn = sqlite3.connect("data/collision_data.db")
    query = "SELECT borough_name, lat_min, lat_max, lon_min, lon_max FROM Boroughs;"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df.set_index("borough_name").T.to_dict()

borough_ranges = get_borough_ranges()
unique_boroughs = list(borough_ranges.keys())

# Initialize session state for borough selection
if "selected_borough" not in st.session_state:
    st.session_state.selected_borough = unique_boroughs[0]  # Default to the first borough

# Rerun mechanism: Fake submit button for updating borough selection
if "force_rerun" not in st.session_state:
    st.session_state.force_rerun = False

# Sidebar for Borough Selection
st.sidebar.header("Borough Selection")
new_borough = st.selectbox("Select Borough", unique_boroughs)

# Trigger a rerun if the borough changes
if new_borough != st.session_state.selected_borough:
    st.session_state.selected_borough = new_borough
    st.session_state.force_rerun = True  # Trigger rerun

# Debugging: Display borough ranges (optional)
print(f"Selected Borough: {st.session_state.selected_borough}")
selected_borough = st.session_state.selected_borough

# Get latitude and longitude range for the selected borough
lat_min = borough_ranges[selected_borough]["lat_min"]
lat_max = borough_ranges[selected_borough]["lat_max"]
lon_min = borough_ranges[selected_borough]["lon_min"]
lon_max = borough_ranges[selected_borough]["lon_max"]

# --- Prediction Form ---
with st.form("prediction_form"):
    st.write("Enter Conditions for Prediction:")

    # Render sliders for latitude and longitude dynamically
    input_latitude = st.slider(
        "Latitude",
        min_value=float(lat_min), max_value=float(lat_max),
        value=(float(lat_min) + float(lat_max)) / 2, step=0.001,
        key=f"latitude_slider_{selected_borough}"
    )
    input_longitude = st.slider(
        "Longitude",
        min_value=float(lon_min), max_value=float(lon_max),
        value=(float(lon_min) + float(lon_max)) / 2, step=0.001,
        key=f"longitude_slider_{selected_borough}"
    )

    # Other form inputs
    input_factor = st.selectbox("Contributing Factor", get_unique_values("contributing_factor_category"))
    input_vehicle = st.selectbox("Vehicle Type", get_unique_values("vehicle_category"))
    input_month = st.selectbox("Month", month_names)
    input_day_of_week = st.selectbox("Day of the Week", get_unique_values("day_of_week"))
    input_time_of_day = st.selectbox("Time of Day", get_unique_values("time_of_day"))

    # Fetch crash rate based on inputs (via SQL)
    conn = sqlite3.connect("data/collision_data.db")
    crash_rate_query = """
    SELECT AVG(crash_rate) as avg_crash_rate
    FROM Collisions
    WHERE borough = ? AND time_of_day = ? AND month = ? AND day_of_week = ?
    """
    avg_crash_rate = pd.read_sql_query(
        crash_rate_query, conn, params=[selected_borough, input_time_of_day, month_names.index(input_month) + 1, input_day_of_week]
    )["avg_crash_rate"].iloc[0]
    conn.close()

    # Submit button
    submitted = st.form_submit_button("Predict")
    if submitted:
        # Prepare input data for the model
        input_data = pd.DataFrame({
            "borough": [selected_borough],
            "time_of_day": [input_time_of_day],
            "contributing_factor_category": [input_factor],
            "vehicle_category": [input_vehicle],
            "crash_rate": [avg_crash_rate],
            "latitude": [input_latitude],
            "longitude": [input_longitude],
            "month": [month_names.index(input_month) + 1],
            "day_of_week": [input_day_of_week],
        })
        # Create aligned input using feature names for the prediction
        aligned_input = create_aligned_input(input_data, feature_names)
        input_processed = aligned_input.values
        prediction = rf_model.predict(input_processed)
        prediction_label = label_encoder.inverse_transform(prediction)
        st.write(f"Predicted Severity Category: **{prediction_label[0]}**")
    
# Hidden button to force rerun
if st.session_state.force_rerun:
    st.button("lat, long ranges are being set...", on_click=lambda: st.rerun())