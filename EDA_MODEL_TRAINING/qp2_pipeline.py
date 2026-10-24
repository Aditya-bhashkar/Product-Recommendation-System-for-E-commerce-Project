# ============================================================
# FOUNDATIONS OF DATA SCIENCE - QP2 CASE STUDY 2
# Product Recommendation System for E-commerce
#
# Pipeline:
# CSV -> PostgreSQL + MongoDB -> Python Extraction
# -> Data Integration -> Preprocessing -> Random Forest
# -> Accuracy, Precision, Recall
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import os

import pandas as pd
import numpy as np
import psycopg2
from pymongo import MongoClient

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    classification_report
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

CSV_FILE = "Product_Recommendation_Cleaned.csv"

POSTGRES_HOST = "localhost"
POSTGRES_PORT = "5432"
POSTGRES_DATABASE = "fds_qp2"
POSTGRES_USER = "postgres"

POSTGRES_PASSWORD = os.environ.get("POSTGRES_PASSWORD")

MONGO_URI = "mongodb://localhost:27017/"
MONGO_DATABASE = "fds_qp2"
MONGO_COLLECTION = "interaction_logs"


# ============================================================
# 2. CSV COLUMN NAMES
# ============================================================

REG_COL = "Registration Number"
AGE_COL = "What is your age group?"
GENDER_COL = "What is your gender?"
OCCUPATION_COL = "What is your occupation"
INCOME_COL = "What is your monthly income range (₹)?"
SHOPPING_FREQ_COL = "How often do you shop online?"
DEVICE_COL = " Which device do you primarily use for online shopping?"
PLATFORM_COL = "Which e-commerce platform do you use most frequently?"
SPENDING_COL = " What is your average monthly spending on online shopping (₹)?"
CATEGORY_COL = " Which product category do you buy most often online?"
FACTOR_COL = "What is the most important factor when choosing a product?"
TIMESTAMP_COL = "Timestamp"
DISCOVERY_COL = " How do you usually discover new products online?"
RECOMMENDATION_FREQ_COL = (
    " How often do you purchase products based on recommendations "
    "shown on the platform?"
)
PAYMENT_COL = "What payment method do you prefer?"
TARGET_COL = (
    "How satisfied are you with product recommendations shown to you "
    "on e-commerce platforms?"
)
RETURN_COL = "Have you ever returned a product purchased online?"


# ============================================================
# 3. READ AND VALIDATE CSV
# ============================================================

print("\n========== STEP 1: READING CSV ==========")

df = pd.read_csv(CSV_FILE)

print("Number of rows:", len(df))
print("Number of columns:", len(df.columns))

required_columns = [
    REG_COL,
    AGE_COL,
    GENDER_COL,
    OCCUPATION_COL,
    INCOME_COL,
    SHOPPING_FREQ_COL,
    DEVICE_COL,
    PLATFORM_COL,
    SPENDING_COL,
    CATEGORY_COL,
    FACTOR_COL,
    TIMESTAMP_COL,
    DISCOVERY_COL,
    RECOMMENDATION_FREQ_COL,
    PAYMENT_COL,
    TARGET_COL,
    RETURN_COL
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    print("\nERROR: The following columns are missing:")
    for column in missing_columns:
        print("-", repr(column))
    raise ValueError("CSV column names do not match the pipeline.")

if df[REG_COL].isnull().any():
    raise ValueError("Registration Number contains missing values.")

if df[REG_COL].duplicated().any():
    raise ValueError("Duplicate Registration Numbers found.")

if df[TARGET_COL].isnull().any():
    raise ValueError("Target column contains missing values.")

print("CSV validation successful.")
print("Unique Registration Numbers:", df[REG_COL].nunique())


# ============================================================
# 4. PREPARE SIMPLE COLUMN NAMES
# ============================================================

print("\n========== STEP 2: PREPARING DATA ==========")

# Rename the original long column names to shorter names.
rename_map = {
    REG_COL: "registration_number",
    AGE_COL: "age_group",
    GENDER_COL: "gender",
    OCCUPATION_COL: "occupation",
    INCOME_COL: "income_range",
    SHOPPING_FREQ_COL: "shopping_frequency",
    DEVICE_COL: "device",
    PLATFORM_COL: "platform",
    SPENDING_COL: "monthly_spending",
    CATEGORY_COL: "product_category",
    FACTOR_COL: "product_factor",
    TIMESTAMP_COL: "timestamp",
    DISCOVERY_COL: "discovery_method",
    RECOMMENDATION_FREQ_COL: "recommendation_frequency",
    PAYMENT_COL: "payment_method",
    TARGET_COL: "satisfaction",
    RETURN_COL: "returned_product"
}

clean_df = df.rename(columns=rename_map).copy()

# Remove email if it exists.
if "Email Address" in clean_df.columns:
    clean_df = clean_df.drop(columns=["Email Address"])

# Convert blank strings into missing values.
clean_df = clean_df.replace(r"^\s*$", np.nan, regex=True)

print("Prepared dataset shape:", clean_df.shape)


# ============================================================
# 5. CONNECT TO POSTGRESQL
# ============================================================

print("\n========== STEP 3: CONNECTING TO POSTGRESQL ==========")

if not POSTGRES_PASSWORD:
    raise RuntimeError("Set the POSTGRES_PASSWORD environment variable before running the pipeline.")

postgres_connection = psycopg2.connect(
    host=POSTGRES_HOST,
    port=POSTGRES_PORT,
    database=POSTGRES_DATABASE,
    user=POSTGRES_USER,
    password=POSTGRES_PASSWORD
)

postgres_cursor = postgres_connection.cursor()

print("Connected to PostgreSQL successfully.")


# ============================================================
# 6. CREATE AND POPULATE SQL TABLE
# ============================================================

print("\n========== STEP 4: INSERTING DATA INTO POSTGRESQL ==========")

postgres_cursor.execute("""
    CREATE TABLE IF NOT EXISTS customer_data (
        registration_number VARCHAR(50) PRIMARY KEY,
        age_group TEXT,
        gender TEXT,
        occupation TEXT,
        income_range TEXT,
        shopping_frequency TEXT,
        device TEXT,
        platform TEXT,
        monthly_spending TEXT,
        product_category TEXT,
        product_factor TEXT,
        returned_product TEXT
    );
""")

# Clear old records so the script can be executed again.
postgres_cursor.execute("DELETE FROM customer_data;")

sql_insert_query = """
    INSERT INTO customer_data (
        registration_number,
        age_group,
        gender,
        occupation,
        income_range,
        shopping_frequency,
        device,
        platform,
        monthly_spending,
        product_category,
        product_factor,
        returned_product
    )
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
"""

for _, row in clean_df.iterrows():
    postgres_cursor.execute(
        sql_insert_query,
        (
            row["registration_number"],
            row["age_group"],
            row["gender"],
            row["occupation"],
            row["income_range"],
            row["shopping_frequency"],
            row["device"],
            row["platform"],
            row["monthly_spending"],
            row["product_category"],
            row["product_factor"],
            row["returned_product"]
        )
    )

postgres_connection.commit()

print("Inserted", len(clean_df), "records into PostgreSQL.")


# ============================================================
# 7. CONNECT TO MONGODB
# ============================================================

print("\n========== STEP 5: CONNECTING TO MONGODB ==========")

mongo_client = MongoClient(MONGO_URI)

# Test MongoDB connection.
mongo_client.admin.command("ping")

mongo_database = mongo_client[MONGO_DATABASE]
mongo_collection = mongo_database[MONGO_COLLECTION]

print("Connected to MongoDB successfully.")


# ============================================================
# 8. CREATE AND POPULATE NOSQL COLLECTION
# ============================================================

print("\n========== STEP 6: INSERTING DATA INTO MONGODB ==========")

# Clear previous interaction logs.
mongo_collection.delete_many({})

mongo_documents = []

for _, row in clean_df.iterrows():

    document = {
        "registration_number": row["registration_number"],
        "timestamp": row["timestamp"],
        "interaction": {
            "discovery_method": row["discovery_method"],
            "recommendation_frequency": row["recommendation_frequency"],
            "payment_method": row["payment_method"]
        },
        "recommendation": {
            "satisfaction": row["satisfaction"]
        }
    }

    mongo_documents.append(document)

if mongo_documents:
    mongo_collection.insert_many(mongo_documents)

print("Inserted", len(mongo_documents), "documents into MongoDB.")


# ============================================================
# 9. EXTRACT DATA FROM POSTGRESQL
# ============================================================

print("\n========== STEP 7: EXTRACTING DATA FROM POSTGRESQL ==========")

sql_query = """
    SELECT
        registration_number,
        age_group,
        gender,
        occupation,
        income_range,
        shopping_frequency,
        device,
        platform,
        monthly_spending,
        product_category,
        product_factor,
        returned_product
    FROM customer_data;
"""

sql_df = pd.read_sql_query(
    sql_query,
    postgres_connection
)

print("SQL extracted rows:", len(sql_df))
print(sql_df.head())


# ============================================================
# 10. EXTRACT DATA FROM MONGODB
# ============================================================

print("\n========== STEP 8: EXTRACTING DATA FROM MONGODB ==========")

mongo_records = list(
    mongo_collection.find(
        {},
        {"_id": 0}
    )
)

mongo_rows = []

for document in mongo_records:

    interaction = document.get("interaction", {})
    recommendation = document.get("recommendation", {})

    mongo_rows.append({
        "registration_number": document.get("registration_number"),
        "timestamp": document.get("timestamp"),
        "discovery_method": interaction.get("discovery_method"),
        "recommendation_frequency": interaction.get(
            "recommendation_frequency"
        ),
        "payment_method": interaction.get("payment_method"),
        "satisfaction": recommendation.get("satisfaction")
    })

mongo_df = pd.DataFrame(mongo_rows)

print("MongoDB extracted rows:", len(mongo_df))
print(mongo_df.head())


# ============================================================
# 11. MERGE SQL AND NOSQL DATA
# ============================================================

print("\n========== STEP 9: MERGING SQL AND NOSQL DATA ==========")

integrated_df = pd.merge(
    sql_df,
    mongo_df,
    on="registration_number",
    how="inner"
)

print("Integrated dataset shape:", integrated_df.shape)

if integrated_df.empty:
    raise ValueError("The SQL and MongoDB datasets could not be merged.")

if integrated_df["registration_number"].duplicated().any():
    raise ValueError("Duplicate registration numbers found after merging.")

print("\nIntegrated dataset preview:")
print(integrated_df.head())

integrated_df.to_csv(
    "QP2_Integrated_Dataset.csv",
    index=False
)

print("\nSaved: QP2_Integrated_Dataset.csv")


# ============================================================
# 12. DATA PREPROCESSING
# ============================================================

print("\n========== STEP 10: DATA PREPROCESSING ==========")

# Convert timestamp into useful numerical features.
integrated_df["timestamp"] = pd.to_datetime(
    integrated_df["timestamp"],
    errors="coerce"
)

integrated_df["timestamp_hour"] = integrated_df["timestamp"].dt.hour
integrated_df["timestamp_day"] = integrated_df["timestamp"].dt.day
integrated_df["timestamp_month"] = integrated_df["timestamp"].dt.month

# Remove the original timestamp after extracting numerical features.
integrated_df = integrated_df.drop(columns=["timestamp"])

# The target is recommendation satisfaction.
target_column = "satisfaction"

# Remove target and integration key from input features.
X = integrated_df.drop(
    columns=[
        target_column,
        "registration_number"
    ]
)

y = integrated_df[target_column]

# Remove any accidental duplicate target column.
if "satisfaction_x" in X.columns:
    X = X.drop(columns=["satisfaction_x"])

if "satisfaction_y" in X.columns:
    X = X.drop(columns=["satisfaction_y"])

# Remove columns that contain only missing values.
all_missing_columns = [
    column for column in X.columns
    if X[column].isna().all()
]

if all_missing_columns:
    X = X.drop(columns=all_missing_columns)

# Identify categorical and numerical columns.
categorical_columns = X.select_dtypes(
    include=["object", "string", "category"]
).columns.tolist()

numerical_columns = X.select_dtypes(
    include=["int64", "int32", "float64", "float32"]
).columns.tolist()

print("Categorical features:", categorical_columns)
print("Numerical features:", numerical_columns)


# ============================================================
# 13. PREPROCESSING PIPELINE
# ============================================================

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)

numerical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            categorical_pipeline,
            categorical_columns
        ),
        (
            "numerical",
            numerical_pipeline,
            numerical_columns
        )
    ]
)


# ============================================================
# 14. TRAIN-TEST SPLIT
# ============================================================

print("\n========== STEP 11: TRAIN-TEST SPLIT ==========")

# An 80:20 stratified split is used.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))


# ============================================================
# 15. RANDOM FOREST MODEL
# ============================================================

print("\n========== STEP 12: TRAINING MODEL ==========")

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

model_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            model
        )
    ]
)

model_pipeline.fit(X_train, y_train)

print("Random Forest model trained successfully.")


# ============================================================
# 16. MODEL EVALUATION
# ============================================================

print("\n========== STEP 13: MODEL EVALUATION ==========")

y_pred = model_pipeline.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

print("\n========== FINAL RESULTS ==========")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# ============================================================
# 17. ACTUAL VS PREDICTED RESULTS
# ============================================================

prediction_results = pd.DataFrame({
    "Actual Satisfaction": y_test.reset_index(drop=True),
    "Predicted Satisfaction": pd.Series(y_pred)
})

print("\n========== ACTUAL VS PREDICTED ==========")
print(prediction_results)

prediction_results.to_csv(
    "QP2_Prediction_Results.csv",
    index=False
)

print("\nSaved: QP2_Prediction_Results.csv")


# ============================================================
# 18. CLOSE DATABASE CONNECTIONS
# ============================================================

postgres_cursor.close()
postgres_connection.close()
mongo_client.close()

print("\n========== PIPELINE COMPLETED SUCCESSFULLY ==========")