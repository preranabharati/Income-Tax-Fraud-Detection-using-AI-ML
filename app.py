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
    def get_tax_slab(income):
        if income <= 300000:
            return 0
        elif 300000 < income <= 600000:
            return 5
        elif 600000 < income <= 900000:
            return 10
        elif 900000 < income <= 1200000:
            return 15
        elif 1200000 < income <= 1500000:
            return 20
        else:
            return 30

    reported_slab = get_tax_slab(reported_income)
    predicted_slab = get_tax_slab(predicted_income)

    if reported_slab != predicted_slab:
        return "Fraud"
    else:
        return "Not Fraud"


def calculate_tax(income):
    if income <= 300000:
        return 0
    elif 300000 < income <= 600000:
        return 0.05 * (income - 300000)
    elif 600000 < income <= 900000:
        return 0.1 * (income - 600000) + 0.05 * 300000
    elif 900000 < income <= 1200000:
        return 0.15 * (income - 900000) + 0.1 * 300000 + 0.05 * 300000
    elif 1200000 < income <= 1500000:
        return 0.20 * (income - 1200000) + 0.15 * 300000 + 0.1 * 300000 + 0.05 * 300000
    else:
        return 0.30 * (income - 1500000) + 0.20 * 300000 + 0.15 * 300000 + 0.1 * 300000 + 0.05 * 300000

def validate_pan_card(pan_card):
    # PAN card should be in the format AAAAA0000A
    if not pan_card[:5].isalpha() or not pan_card[5:9].isdigit() or not pan_card[9].isalpha():
        raise ValueError("Invalid PAN card format. Please enter in the format AAAAA0000A")

def validate_aadhar_bank(account_number):
    # Aadhar and Bank account numbers should have exactly 12 digits
    if not account_number or not account_number.isdigit() or len(account_number) != 12:
        raise ValueError("Invalid number of digits. Please enter a 12-digit number.")

def main():
    st.title("Fraud Detection App")

    st.header("User Input")
    name = st.text_input("Name")
    pan_card = st.text_input("PAN Card")
    aadhar_card = st.text_input("Aadhar Card")
    bank_account_no = st.text_input("Bank Account No")

    # Validate PAN card format
    try:
        validate_pan_card(pan_card)
    except ValueError as e:
        st.error(str(e))

    # Validate Aadhar card and Bank account numbers
    try:
        validate_aadhar_bank(aadhar_card)
        validate_aadhar_bank(bank_account_no)
    except ValueError as e:
        st.error(str(e))
        return

    age = st.slider("Age", 20, 100, 30)
    occupation = st.selectbox("Occupation", ["Salaried", "Self-employed", "Business"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married"])
    children = st.selectbox("Children (Yes/No)", ["No", "Yes"])
    reported_income = st.number_input("Reported Income")

    # Additional inputs
    if children == "Yes":
        educational_expenses = np.random.uniform(1000, 100000)
    else:
        educational_expenses = 0

    if occupation == "Business":
        business_income = np.random.uniform(1000, 100000)
    else:
        business_income = 0

    ii = st.selectbox("Do you have interest income? (Yes/No)", ["No","Yes"])
    if ii == "Yes":
        interest_income = np.random.uniform(1000, 100000)
    else:
        interest_income = 0

    cg = st.selectbox("Do you have Capital Gains? (Yes/No)", ["No", "Yes"])
    if cg == "Yes":
        capital_gains = np.random.uniform(1000, 100000)
    else:
        capital_gains = 0

    other_income = np.random.uniform(1000, 100000)
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

        # Tax calculation
        st.header("Tax Calculation")
        reported_income_tax = calculate_tax(reported_income)
        predicted_income_tax = calculate_tax(predicted_income)
        st.write(f"Tax paid on Reported Income: ₹{reported_income_tax:,.2f}")
        st.write(f"Tax to be paid on Predicted Income: ₹{predicted_income_tax:,.2f}")

        # Classify fraud
        st.header("Fraud Classification")
        fraud_classification = classify_fraud(reported_income, predicted_income)
        st.write(f"The input is classified as: {fraud_classification}")

if __name__ == "__main__":
    main()
