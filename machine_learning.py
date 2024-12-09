"""
--- Machine Learning: Collision Severity Prediction Modeling ---
For this part we are aiming to predict the severity of motor vehicle collisions. 
The process involves data preparation, model training, evaluation, and saving the trained models.
The key steps are as follows:
1. **Load Data**: Retrieve collision data from a SQLite database.
2. **Target Encoding**: Encode the severity categories (High, Medium, Low) into numerical values.
3. **Feature Selection**: Define features for the model, including categorical and numerical variables.
4. **Data Preprocessing**: Standardize numerical features and one-hot encode categorical variables.
5. **Handle Class Imbalance**: Use SMOTE (Synthetic Minority Oversampling Technique) to balance the target variable classes.
6. **Train-Test Split**: Split the data into training and testing sets.
7. **Model Training**: Train two machine learning models:
    - Random Forest
    - XGBoost
8. **Evaluation**: Evaluate both models using classification reports and confusion matrices.
9. **Save Models**: Save the trained models and label encoder for future use.

--- Observation & Evaluation --- 

"""

# Import necessary libraries
import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import joblib

# Step 1: Load data from the SQLite database
print("\n--- Loading Data from Database ---")
db_file = "data/collision_data.db"
conn = sqlite3.connect(db_file)

# Query to retrieve relevant data
query = """
SELECT 
    severity_category, crash_rate, day_of_week, time_of_day, month, 
    vehicle_category, contributing_factor_category, 
    collision_cluster, borough, latitude, longitude 
FROM Collisions
"""
df = pd.read_sql_query(query, conn)
conn.close()

# Step 2: Encode target variable
print("\n--- Encoding Severity Category ---")
# Encode severity categories into numerical labels (High, Medium, Low -> 0, 1, 2)
label_encoder = LabelEncoder()
df['severity_category_encoded'] = label_encoder.fit_transform(df['severity_category'])

# Step 3: Define features and target variable
# Features (X) exclude the original and encoded target variables
X = df.drop(columns=['severity_category', 'severity_category_encoded'])
# Target variable (y) is the encoded severity category
y = df['severity_category_encoded']

# Step 4: Define categorical and numerical features
categorical_features = ['day_of_week', 'time_of_day', 'month', 'vehicle_category', 'contributing_factor_category', 'borough']
numerical_features = ['latitude', 'longitude', 'crash_rate']

# Step 5: Preprocessor for numerical and categorical data
print("\n--- Defining Preprocessor ---")
# Standardize numerical data and one-hot encode categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

# Step 6: Preprocess data before applying SMOTE
print("\n--- Preprocessing Data for SMOTE ---")

# Transform the data using the preprocessor
X_preprocessed = preprocessor.fit_transform(X)

# Get the feature names
feature_names = preprocessor.get_feature_names_out()

# Step 7: Handle class imbalance with SMOTE
print("\n--- Handling Class Imbalance with SMOTE ---")
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_preprocessed, y)

# Step 8: Split the data into training and testing sets
print("\n--- Splitting Data into Training and Testing Sets ---")
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Step 9: Train a Random Forest model
print("\n--- Training Random Forest for Collision Risk Prediction ---")
# Create a pipeline for Random Forest
rf_pipeline = Pipeline([
    ('classifier', RandomForestClassifier(class_weight="balanced", random_state=42))
])
rf_pipeline.fit(X_train, y_train)  # Train the Random Forest model
y_pred_rf = rf_pipeline.predict(X_test)  # Make predictions on the test set

# Decode the target variable for evaluation
y_test_decoded = label_encoder.inverse_transform(y_test)
y_pred_rf_decoded = label_encoder.inverse_transform(y_pred_rf)

# Evaluate the Random Forest model
print("Random Forest Classification Report:")
print(classification_report(y_test_decoded, y_pred_rf_decoded, zero_division=1))
print("Confusion Matrix:")
print(confusion_matrix(y_test_decoded, y_pred_rf_decoded))

# Step 10: Train an XGBoost model
print("\n--- Training XGBoost for Collision Severity Prediction ---")
# Create a pipeline for XGBoost
xgb_pipeline = Pipeline([
    ('classifier', XGBClassifier(
        scale_pos_weight=1, random_state=42, use_label_encoder=False, eval_metric='mlogloss'
    ))
])
xgb_pipeline.fit(X_train, y_train)  # Train the XGBoost model
y_pred_xgb = xgb_pipeline.predict(X_test)  # Make predictions on the test set

# Decode the target variable for evaluation
y_pred_xgb_decoded = label_encoder.inverse_transform(y_pred_xgb)

# Evaluate the XGBoost model
print("XGBoost Classification Report:")
print(classification_report(y_test_decoded, y_pred_xgb_decoded, zero_division=1))
print("Confusion Matrix:")
print(confusion_matrix(y_test_decoded, y_pred_xgb_decoded))

# Step 11: Save trained models and encoder
print("\n--- Saving Models, Encoder and preprocessor ---")
joblib.dump(rf_pipeline, "models/random_forest_model.pkl")  # Save Random Forest model
joblib.dump(xgb_pipeline, "models/xgboost_model.pkl")  # Save XGBoost model
joblib.dump(label_encoder, "encoders/label_encoder.pkl")  # Save Label Encoder

# Save the feature names
with open("models/feature_names.txt", "w") as f:
    f.write("\n".join(feature_names))
