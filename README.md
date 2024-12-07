### Motor Vehicle Collision Severity Prediction in NYC

**Project Definition:**  
This project aims to develop a predictive model for motor vehicle collisions in NYC, aiming to identify high-risk zones and assess the severity of incidents. By analyzing historical collision data, the model can help public safety and transportation officials in their preventive measures to reduce accidents and make their city safer for vehicle motorists.

---

### Requirements

The following libraries are required to run this project. Use the `requirements.txt` file to install dependencies:

```bash
pip install -r requirements.txt
# Alternatively, if pip fails:
pip3 install -r requirements.txt
```

### [IMPORTANT] Additional Files [REQUIRED]

Some required files are excluded from the repository due to size constraints. Use one of the following methods to retrieve them:

1. **Run the provided script:**
   ```bash
   python download_files.py
   # Alternatively:
   python3 download_files.py
   ```

2. **Manual Download:**
   - Save **`raw_dataset.csv`** to the `data/` folder: [Download Link](https://drive.google.com/uc?id=15QJIa6AxucXFIwCEITNL_uioh8cEj8Er)
   - Save **`clean_collision_data.csv`** to the `data/` folder: [Download Link](https://drive.google.com/uc?id=12YVvDoTXSMhq65jYXiDpLW0pvLY-J8M1)
   - Save **`random_forest_model.pkl`** to the `models/` folder: [Download Link](https://drive.google.com/uc?id=1dwqypIt_eLYZM1tM8UR7g7HBuwxy17df)

---

### Part 1: Data Ingestion and Cleaning  
**Script:** `data_cleaning.py`  

**Purpose:**  
Load and clean raw data to ensure it is consistent and usable for downstream processes.  

**Key Actions:**  
- Loaded raw data using pandas.  
- Addressed missing values, removed duplicates, and standardized data formats.  
- Retained critical fields for analysis while discarding redundant or incomplete columns.  

**Result:**  
Generated a structured, clean dataset optimized for analysis and feature engineering.

---

### Part 2: Exploratory Data Analysis (EDA)  
**Script:** `exploratory_data_analysis.py`  

**Purpose:**  
Investigate the dataset to identify essential features and patterns for modeling.  

**Key Actions:**  
- Visualized data distributions for numerical and categorical features using seaborn.  
- Investigated correlations between features and collision severity.  
- Analyzed trends, such as time-of-day patterns and spatial distributions.  

**Result:**  
Gained insights to guide dataset cleaning, feature selection, and preprocessing.

---

### Part 3: Feature Engineering  
**Script:** `feature_engineering.py`  

**Purpose:**  
Transform raw data into meaningful features to enhance model performance and interpretability.  

**Key Actions:**  
- **Severity Score:** Weighted injuries and fatalities to differentiate collision impacts.  
- **Temporal Features:** Extracted day of the week, time of day, and month.  
- **Spatial Features:** Incorporated location grids and collision clustering (DBSCAN).  
- **Categorical Simplification:** Grouped vehicle types and contributing factors into broader, interpretable categories.  
- **Class Imbalance Handling:** Addressed imbalance through downsampling and SMOTE.  

**Result:**  
Produced a feature-rich dataset tailored for machine learning models.

---

### Part 4: Database Integration  
**Script:** `database_integration.py`  

**Purpose:**  
Store collision data efficiently for querying and dynamic updates.  

**Key Actions:**  
- Designed and implemented relational tables for collisions, boroughs, and statistics.  
- Loaded processed data into SQLite for efficient storage and indexing.  
- Integrated SQL queries for statistical computation in the Streamlit application.  

**Result:**  
Built a robust backend for seamless interaction with the user interface.

---

### Part 5: Machine Learning  
**Script:** `machine_learning.py`  

**Purpose:**  
Train models to predict collision severity based on engineered features.  

**Key Actions:**  
- **Model Training:** Used Random Forest and XGBoost classifiers.  
- **Pipeline Construction:** Included feature scaling, one-hot encoding, and SMOTE.  
- **Evaluation:** Used precision, recall, F1-score, and confusion matrices to assess performance.  
- **Model Selection:** Chose Random Forest as the best-performing model.  

**Result:**  
Developed an accurate and interpretable model for collision severity prediction.

---

### Part 6: Interactive Interface  
**Script:** `app.py`  

**Purpose:**  
Provide a user-friendly dashboard for dynamic analysis and predictions.  

**Key Actions:**  
- **Visualization:** Displayed heatmaps of collision hotspots using folium.  
- **Prediction:** Allowed users to input parameters (e.g., borough, time of day) and receive severity predictions.  
- **Dynamic Updates:** Integrated SQL queries for real-time data retrieval and statistics.  

**Result:**  
Created an accessible tool for exploring collision data and generating predictions.

---

### How to Run the Application:

1. Navigate to the project directory.  
2. Run the following command:  
   ```bash
   streamlit run app.py
   ```
3. If the webpage doesn't open automatically, use the local URL provided in the terminal.  
---
### Execution Steps from start to finish (May Take A While For Each Step):
1. Ensure you have the required file from [REQUIRED FILES](#important-additional-files-required) and that `data/collision_data.db` is deleted.
2. Run Script `data_cleaning.py`
3. Run Script `feature_engineering.py`
4. Run Script `data_integration.py`
5. Run Script `machine_learning.py`
5. Run Script `machine_learning.py`
6. Run Command In Terminal: `streamlit run app.py`

### Conclusion

This project showcases the integration of data engineering, machine learning, and visualization to address real-world challenges in urban safety. By predicting collision severity, it offers actionable insights for urban planners, safety officials, and the public. Future improvements could include real-time data integration and expanding the model to predict collision likelihood alongside severity.

---

### Further Analysis Conducted  

**Extended EDA (Features Analysis):**  
- **Severity Score Analysis:** Assessed patterns in collision impacts.  
- **Threshold Optimization:** Defined boundaries for severity categories (Low, Medium, High).  
- **Class Imbalance Resolution:** Balanced severity categories through downsampling and SMOTE.  

**Models and Encoders:**  
- Stored machine learning models (`random_forest_model.pkl`, `xgboost_model.pkl`) and preprocessing encoders for reuse.  
- Accessible through the Streamlit application for predictions.  
