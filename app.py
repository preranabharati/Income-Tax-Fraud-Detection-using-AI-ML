import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the label encoders
label_encoder_occupation = joblib.load("label_encoder_occupation.joblib")
label_encoder_marital = joblib.load("label_encoder_marital_status.joblib")
label_encoder_children = joblib.load("label_encoder_children.joblib")

def load_best_model():
    return joblib.load("best_model.joblib")

def predict_income(model, input_data):
    return model.predict(input_data.reshape(1, -1))[0]

def classify_fraud(reported_income, predicted_income):
    threshold_percentage = 0.1 # Adjust the threshold based on the model's performance
    percentage_difference = abs((reported_income - predicted_income) / reported_income)
    if percentage_difference > threshold_percentage:
        return "Fraud"
    else:
        return "Not Fraud"

def main():
    st.title("Fraud Detection App")

    st.header("User Input")
    name = st.text_input("Name")
    pan_card = st.text_input("PAN Card")
    aadhar_card = st.text_input("Aadhar Card")
    bank_account_no = st.text_input("Bank Account No")
	
    age = st.slider("Age", 20, 100, 30)
    occupation = st.selectbox("Occupation", ["Salaried", "Self-employed", "Business"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married"])
    children = st.selectbox("Children (Yes/No)", ["No", "Yes"])
    reported_income = st.number_input("Reported Income")
    interest_income = np.random.uniform(1000, 100000)
    business_income = np.random.uniform(1000, 100000)
    capital_gains = np.random.uniform(1000, 100000)
    other_income = np.random.uniform(1000, 100000)
    educational_expenses = np.random.uniform(1000, 100000)
    healthcare_costs = np.random.uniform(1000, 100000)
    lifestyle_expenditure = np.random.uniform(1000, 100000)
    other_expenses = np.random.uniform(1000, 100000)
    bank_debited = np.random.uniform(1000, 100000)
    credit_card_debited = np.random.uniform(1000, 100000)

    if st.button("Detect Fraud"):
        # Create DataFrame without considering Name, PAN Card, Aadhar Card, and Bank Account Number
        input_data = pd.DataFrame({
            "Age": [age],
            "Occupation": [occupation],
            "Marital Status": [marital_status],
            "Children (Yes/No)": [children],
            "Reported Income": [reported_income],
            "Interest Income": [interest_income],
            "Business Income": [business_income],
            "Capital Gains": [capital_gains],
            "Other Income": [other_income],
            "Educational Expenses": [educational_expenses],
            "Healthcare Costs": [healthcare_costs],
            "Lifestyle Expenditure": [lifestyle_expenditure],
            "Other Expenses": [other_expenses],
            "Bank Debited": [bank_debited],
            "Credit Card Debited": [credit_card_debited]
        })

        # Encode categorical variables
        input_data["Occupation"] = label_encoder_occupation.transform([input_data["Occupation"].values[0]])[0]
        input_data["Marital Status"] = label_encoder_marital.transform([input_data["Marital Status"].values[0]])[0]
        input_data["Children (Yes/No)"] = label_encoder_children.transform([input_data["Children (Yes/No)"].values[0]])[0]

        # Load the best model
        best_model = load_best_model()

        # Predict income
        st.header("Predicted Income")
        predicted_income = predict_income(best_model, input_data.values.flatten())
        st.write(f"Predicted Income: ₹{predicted_income:,.2f}")

        # Classify fraud
        st.header("Fraud Classification")
        fraud_classification = classify_fraud(reported_income, predicted_income)
        st.write(f"The input is classified as: {fraud_classification}")

if __name__ == "__main__":
    main()