import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import xgboost as xgb

# Load the trained XGBoost model
model = joblib.load('xgboost_model.pkl')

# Load the dataset
file_path = r"C:\\Users\\Kavya_Kavya\\Downloads\\shopping_trends (1).csv"  # Update path if necessary
data_df = pd.read_csv(file_path)

# Prepare data
target_variable = 'Category'
X = data_df.drop(columns=[target_variable, 'Customer ID'], errors='ignore')  # Exclude Customer ID if present
y = data_df[target_variable]

# Encode categorical features
label_encoders = {}
for column in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[column] = le.fit_transform(X[column])
    label_encoders[column] = le

# Function to make predictions
def predict_category(new_data):
    new_data_encoded = new_data.copy()
    for column in new_data_encoded.select_dtypes(include=['object']).columns:
        if column in label_encoders:
            le = label_encoders[column]
            new_data_encoded[column] = le.transform(new_data_encoded[column])
        else:
            st.warning(f"Column {column} not found in label encoders.")
            return None
    return model.predict(new_data_encoded)

# Streamlit Interface
st.title("Inventory Management and Sales Prediction App")

# Input features for prediction
st.header("Predict Category for New Data")
input_data = {}
for column in X.columns:
    input_data[column] = st.text_input(f"Enter value for {column}", "")

# Convert input to DataFrame for prediction
if st.button("Predict"):
    input_df = pd.DataFrame([input_data])
    prediction = predict_category(input_df)
    if prediction is not None:
        st.success(f"Predicted Category: {prediction[0]}")

# Display inventory metrics
st.header("Inventory Metrics Dashboard")
st.write("Inventory Turnover Rate: TBD")
st.write("Stockout Rate: TBD")
st.write("Average Sales per Category: TBD")

# Placeholder for alerts
st.header("Automated Alerts")
threshold = st.slider("Set Trend Prediction Threshold", 0.0, 1.0, 0.6)
predicted_probabilities = model.predict_proba(X)

# Check for high-confidence predictions
high_confidence_alerts = [i for i, prob in enumerate(predicted_probabilities) if max(prob) > threshold]

if st.button("Check for High Confidence Alerts"):
    if high_confidence_alerts:
        st.warning(f"High confidence predictions for indexes: {high_confidence_alerts}")
    else:
        st.success("No high confidence predictions.")

# Feedback Loop
st.header("Customer Feedback Loop")
feedback = st.text_area("Enter Customer Feedback", "")
if st.button("Submit Feedback"):
    st.success("Feedback submitted successfully!")

# Visualize prediction distribution
st.header("Prediction Distribution")
if st.button("Show Prediction Distribution"):
    plt.figure(figsize=(10, 5))
    plt.hist(predicted_probabilities.max(axis=1), bins=10, edgecolor='black')
    plt.title("Distribution of Prediction Probabilities")
    plt.xlabel("Prediction Probability")
    plt.ylabel("Frequency")
    st.pyplot(plt)
