import streamlit as st
import pandas as pd
import joblib

# Loading the saved model and other files
model = joblib.load("churn_stacking_model.pkl")

# Preprocessing
df = pd.read_csv('Telco-Customer-Churn.csv')

# Removing it as it is just an identifier
X = df.drop(['customerID', 'Churn'], axis=1)

y = df['Churn'].map({'No':0, 'Yes':1}) # Important to convert to 0-1

# Converting total charges to numeric
X['TotalCharges'] = pd.to_numeric(X['TotalCharges'], errors='coerce')
X['TotalCharges'] = X['TotalCharges'].fillna(X['TotalCharges'].median())

# Encoding the categorical features
X = pd.get_dummies(X, drop_first=True)

feature_columns = X.columns.tolist()

st.title("Customer Churn Predictor")
st.write("Enter customer details to predict churn risk.")

# Taking inputs from the user

gender = st.selectbox("Gender", ["Female", "Male"])

senior = st.selectbox("Senior Citizen", ["No", "Yes"])

partner = st.selectbox("Partner", ["No", "Yes"])

dependents = st.selectbox("Dependents", ["No", "Yes"])

tenure = st.slider("Tenure (months)", 0, 72, 12)

phone_service = st.selectbox("Phone Service", ["No", "Yes"])

multiple_lines = st.selectbox(
    "Multiple Lines",
    ["No", "Yes", "No phone service"]
)

internet_service = st.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

online_security = st.selectbox(
    "Online Security",
    ["No", "Yes", "No internet service"]
)

online_backup = st.selectbox(
    "Online Backup",
    ["No", "Yes", "No internet service"]
)

device_protection = st.selectbox(
    "Device Protection",
    ["No", "Yes", "No internet service"]
)

tech_support = st.selectbox(
    "Tech Support",
    ["No", "Yes", "No internet service"]
)

streaming_tv = st.selectbox(
    "Streaming TV",
    ["No", "Yes", "No internet service"]
)

streaming_movies = st.selectbox(
    "Streaming Movies",
    ["No", "Yes", "No internet service"]
)

contract = st.selectbox(
    "Contract Type",
    ["Month-to-month", "One year", "Two year"]
)

paperless = st.selectbox(
    "Paperless Billing",
    ["No", "Yes"]
)

payment_method = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

monthly_charges = st.slider(
    "Monthly Charges ($)",
    0.0,
    150.0,
    70.0
)

total_charges = st.number_input(
    "Total Charges ($)",
    0.0,
    10000.0,
    1000.0
)

# Predict button

if st.button("Predict Churn"):

    # Making a dictionary with every feature as 0 first
    input_dict = {col: 0 for col in feature_columns}

    # Filling normal number values
    input_dict["tenure"] = tenure
    input_dict["MonthlyCharges"] = monthly_charges
    input_dict["TotalCharges"] = total_charges

    # Yes/No values become 1 or 0
    input_dict["SeniorCitizen"] = 1 if senior == "Yes" else 0
    
    # For these columns, only "Yes" exists because drop_first=True was used
    if partner == "Yes":
        input_dict["Partner_Yes"] = 1

    if dependents == "Yes":
        input_dict["Dependents_Yes"] = 1

    if phone_service == "Yes":
        input_dict["PhoneService_Yes"] = 1

    if paperless == "Yes":
        input_dict["PaperlessBilling_Yes"] = 1


    # If user selected Male then make this column 1
    if gender == "Male":
        input_dict["gender_Male"] = 1

    # For categorical values, only the selected option becomes 1

    if f"MultipleLines_{multiple_lines}" in input_dict:
        input_dict[f"MultipleLines_{multiple_lines}"] = 1

    if f"InternetService_{internet_service}" in input_dict:
        input_dict[f"InternetService_{internet_service}"] = 1

    if f"OnlineSecurity_{online_security}" in input_dict:
        input_dict[f"OnlineSecurity_{online_security}"] = 1

    if f"OnlineBackup_{online_backup}" in input_dict:
        input_dict[f"OnlineBackup_{online_backup}"] = 1

    if f"DeviceProtection_{device_protection}" in input_dict:
        input_dict[f"DeviceProtection_{device_protection}"] = 1

    if f"TechSupport_{tech_support}" in input_dict:
        input_dict[f"TechSupport_{tech_support}"] = 1

    if f"StreamingTV_{streaming_tv}" in input_dict:
        input_dict[f"StreamingTV_{streaming_tv}"] = 1

    if f"StreamingMovies_{streaming_movies}" in input_dict:
        input_dict[f"StreamingMovies_{streaming_movies}"] = 1

    if f"Contract_{contract}" in input_dict:
        input_dict[f"Contract_{contract}"] = 1

    if f"PaymentMethod_{payment_method}" in input_dict:
        input_dict[f"PaymentMethod_{payment_method}"] = 1

    # Converting dictionary into a dataframe because model takes dataframe as input
    input_df = pd.DataFrame([input_dict])

    # Keeping columns in the same order as during training
    input_df = input_df[feature_columns]

    # Use this only if the model was trained after scaling these columns
    # numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    # input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

    # Predicting whether customer will churn or not
    prediction = model.predict(input_df)[0]

    # Getting the probability of churn
    probability = model.predict_proba(input_df)[0][1]

    # Showing the final result
    if prediction == 1:
        st.error(f"High churn risk — {probability:.1%} probability")
    else:
        st.success(f"Low churn risk — {probability:.1%} probability")