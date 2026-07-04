from flask import Flask, render_template, request
import os
import pickle
import numpy as np

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "loan_model.pkl")

with open(MODEL_PATH, "rb") as model_file:
    model = pickle.load(model_file)


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

    features = np.array([[
        1 if gender == "Male" else 0,
        1 if married == "Yes" else 0,
        int(dependents),
        1 if education == "Graduate" else 0,
        1 if self_employed == "Yes" else 0,
        float(applicant_income),
        float(coapplicant_income),
        float(loan_amount),
        float(loan_amount_term),
        int(credit_history),
        {"Rural": 0, "Semiurban": 1, "Urban": 2}[property_area]
    ]])

    prediction = model.predict(features)[0]

    return render_template(
        "result.html",
        prediction=prediction,
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
        property_area=property_area
    )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))