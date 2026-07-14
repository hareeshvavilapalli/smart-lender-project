from flask import Flask, render_template, request
import os
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
print("Model Loaded Successfully")


@app.route('/')
def home():
    return render_template("home.html")


@app.route('/form')
def form():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():
    name = request.form.get("Name", "")
    email = request.form.get("Email", "")
    phone = request.form.get("Phone", "")

    gender = request.form.get("Gender", "")
    married = request.form.get("Married", "")
    dependents = request.form.get("Dependents", "0")
    education = request.form.get("Education", "")
    self_employed = request.form.get("Self_Employed", "")

    applicant_income = request.form.get("ApplicantIncome", "0")
    coapplicant_income = request.form.get("CoapplicantIncome", "0")
    loan_amount = request.form.get("LoanAmount", "0")
    loan_amount_term = request.form.get("Loan_Amount_Term", "0")
    credit_history = request.form.get("Credit_History", "0")
    property_area = request.form.get("Property_Area", "Rural")

    print("request.form:", request.form)

    required_fields = {
        "Gender": gender,
        "Married": married,
        "Education": education,
        "Self_Employed": self_employed,
        "Dependents": dependents,
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_amount_term,
        "Credit_History": credit_history,
        "Property_Area": property_area,
    }
    print("Required fields received:", required_fields)

    if any(value in {"", None} for value in required_fields.values()):
        raise ValueError("One or more required form fields are empty")

    dependents_value = 3 if dependents == "3+" else int(dependents or 0)
    applicant_income_value = float(applicant_income or 0)
    coapplicant_income_value = float(coapplicant_income or 0)
    loan_amount_value = float(loan_amount or 0)
    loan_amount_term_value = float(loan_amount_term or 0)
    credit_history_value = int(float(credit_history or 0))

    features = [
        1 if gender == "Male" else 0,
        1 if married == "Yes" else 0,
        float(dependents_value),
        1 if education == "Graduate" else 0,
        1 if self_employed == "Yes" else 0,
        applicant_income_value,
        coapplicant_income_value,
        loan_amount_value,
        loan_amount_term_value,
        float(credit_history_value),
        {"Rural": 0, "Semiurban": 1, "Urban": 2}[property_area],
    ]

    print("Feature vector before scaling:", features)
    print("Numeric check:", all(isinstance(value, (int, float, np.floating)) for value in features))

    feature_df = pd.DataFrame([features], columns=[
        "Gender",
        "Married",
        "Dependents",
        "Education",
        "Self_Employed",
        "ApplicantIncome",
        "CoapplicantIncome",
        "LoanAmount",
        "Loan_Amount_Term",
        "Credit_History",
        "Property_Area",
    ])
    scaled_features = scaler.transform(feature_df)
    print("Scaled feature vector:", scaled_features)

    prediction_prob = float(model.predict_proba(scaled_features)[0, 1])
    prediction = int(prediction_prob >= 0.5)
    print("Prediction probability:", prediction_prob)
    print("Prediction:", prediction)

    prediction_text = "Loan Approved" if prediction == 1 else "Loan Not Approved"

    return render_template(
        "result.html",
        prediction=prediction,
        prediction_prob=prediction_prob,
        prediction_text=prediction_text,
        name=name,
        email=email,
        phone=phone,
        gender=gender,
        married=married,
        dependents=dependents,
        education=education,
        self_employed=self_employed,
        applicant_income=applicant_income,
        coapplicant_income=coapplicant_income,
        loan_amount=loan_amount,
        loan_amount_term=loan_amount_term,
        credit_history=credit_history,
        property_area=property_area,
    )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))