from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR.parent / "Dataset" / "loan_prediction.csv"
MODEL_PATH = BASE_DIR / "model.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"
LEGACY_MODEL_PATH = BASE_DIR / "loan_model.pkl"

FEATURE_COLUMNS = [
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
]


def encode_dataset(df: pd.DataFrame) -> pd.DataFrame:
    encoded = df.copy()

    encoded["Gender"] = encoded["Gender"].fillna("Male").map({"Male": 1, "Female": 0}).astype(float)
    encoded["Married"] = encoded["Married"].fillna("Yes").map({"Yes": 1, "No": 0}).astype(float)
    encoded["Education"] = encoded["Education"].fillna("Graduate").map({"Graduate": 1, "Not Graduate": 0}).astype(float)
    encoded["Self_Employed"] = encoded["Self_Employed"].fillna("No").map({"Yes": 1, "No": 0}).astype(float)

    encoded["Dependents"] = encoded["Dependents"].fillna("0").replace({"3+": "3"})
    encoded["Dependents"] = encoded["Dependents"].astype(int).astype(float)

    encoded["Property_Area"] = encoded["Property_Area"].fillna("Rural").map({"Rural": 0, "Semiurban": 1, "Urban": 2}).astype(float)
    encoded["Credit_History"] = pd.to_numeric(encoded["Credit_History"], errors="coerce").fillna(1).astype(float)

    numeric_columns = [
        "ApplicantIncome",
        "CoapplicantIncome",
        "LoanAmount",
        "Loan_Amount_Term",
    ]
    for column in numeric_columns:
        encoded[column] = pd.to_numeric(encoded[column], errors="coerce")
        encoded[column] = encoded[column].fillna(encoded[column].median()).astype(float)

    encoded["Loan_Status"] = encoded["Loan_Status"].fillna("N").map({"Y": 1, "N": 0}).astype(int)
    return encoded


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    print("Loaded dataset with", len(df), "rows")
    print("Loan_Status value counts:")
    print(df["Loan_Status"].value_counts(dropna=False))

    encoded_df = encode_dataset(df)
    X = encoded_df[FEATURE_COLUMNS]
    y = encoded_df["Loan_Status"]

    if len(np.unique(y)) < 2:
        raise ValueError("The training data must contain both approved and not-approved labels.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train_scaled, y_train)

    predictions = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, predictions)

    print("Accuracy:", round(accuracy, 4))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))
    print("Classification Report:")
    print(classification_report(y_test, predictions))

    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(model, LEGACY_MODEL_PATH)

    sample = {
        "Gender": "Male",
        "Married": "Yes",
        "Dependents": "0",
        "Education": "Graduate",
        "Self_Employed": "No",
        "ApplicantIncome": 8000,
        "CoapplicantIncome": 2500,
        "LoanAmount": 120,
        "Loan_Amount_Term": 360,
        "Credit_History": 1,
        "Property_Area": "Urban",
    }
    sample_features = [
        1 if sample["Gender"] == "Male" else 0,
        1 if sample["Married"] == "Yes" else 0,
        0 if sample["Dependents"] == "0" else 1 if sample["Dependents"] == "1" else 2 if sample["Dependents"] == "2" else 3,
        1 if sample["Education"] == "Graduate" else 0,
        1 if sample["Self_Employed"] == "Yes" else 0,
        float(sample["ApplicantIncome"]),
        float(sample["CoapplicantIncome"]),
        float(sample["LoanAmount"]),
        float(sample["Loan_Amount_Term"]),
        float(sample["Credit_History"]),
        {"Rural": 0, "Semiurban": 1, "Urban": 2}[sample["Property_Area"]],
    ]
    sample_df = pd.DataFrame([sample_features], columns=FEATURE_COLUMNS)
    sample_scaled = scaler.transform(sample_df)
    sample_prediction = int(model.predict(sample_scaled)[0])
    print("Test applicant input:", sample)
    print("Test applicant features:", sample_features)
    print("Test applicant prediction:", sample_prediction)
    print("Test applicant prediction text:", "Loan Approved" if sample_prediction == 1 else "Loan Not Approved")

    print("Artifacts saved to", MODEL_PATH, SCALER_PATH)


if __name__ == "__main__":
    main()
