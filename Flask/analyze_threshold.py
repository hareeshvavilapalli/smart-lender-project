import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_curve

model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')
df = pd.read_csv('../Dataset/loan_prediction.csv')

encoded = df.copy()
encoded['Gender'] = encoded['Gender'].fillna('Male').map({'Male': 1, 'Female': 0}).astype(float)
encoded['Married'] = encoded['Married'].fillna('Yes').map({'Yes': 1, 'No': 0}).astype(float)
encoded['Education'] = encoded['Education'].fillna('Graduate').map({'Graduate': 1, 'Not Graduate': 0}).astype(float)
encoded['Self_Employed'] = encoded['Self_Employed'].fillna('No').map({'Yes': 1, 'No': 0}).astype(float)
encoded['Dependents'] = encoded['Dependents'].fillna('0').replace({'3+': '3'}).astype(int).astype(float)
encoded['Property_Area'] = encoded['Property_Area'].fillna('Rural').map({'Rural': 0, 'Semiurban': 1, 'Urban': 2}).astype(float)
encoded['Credit_History'] = pd.to_numeric(encoded['Credit_History'], errors='coerce').fillna(1).astype(float)
for col in ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']:
    encoded[col] = pd.to_numeric(encoded[col], errors='coerce').fillna(encoded[col].median()).astype(float)
encoded['Loan_Status'] = encoded['Loan_Status'].fillna('N').map({'Y': 1, 'N': 0}).astype(int)

features = ['Gender','Married','Dependents','Education','Self_Employed','ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History','Property_Area']
X = encoded[features]
y = encoded['Loan_Status']
X_scaled = scaler.transform(X)
proba = model.predict_proba(X_scaled)[:, 1]

precision, recall, thresholds = precision_recall_curve(y, proba)
if len(thresholds) > 0:
    f1 = np.where((precision + recall) > 0, 2 * precision * recall / (precision + recall), 0)
    idx = np.argmax(f1)
    best_threshold = thresholds[idx]
    print('Best threshold by F1:', round(float(best_threshold), 4))
    print('F1:', round(float(f1[idx]), 4))
    print('Precision:', round(float(precision[idx]), 4))
    print('Recall:', round(float(recall[idx]), 4))
else:
    print('No thresholds computed')

for name, feat in [
    ('Good profile', [1, 1, 0, 1, 0, 8000, 2500, 120, 360, 1, 2]),
    ('Lower income but good credit', [1, 1, 0, 1, 0, 5000, 1000, 90, 300, 1, 1]),
    ('High income but bad credit', [1, 1, 0, 1, 0, 12000, 3000, 180, 360, 0, 2]),
    ('Average profile', [0, 1, 1, 1, 0, 6000, 1000, 110, 360, 1, 1]),
]:
    sample_df = pd.DataFrame([feat], columns=features)
    sample_scaled = scaler.transform(sample_df)
    p = model.predict_proba(sample_scaled)[0, 1]
    print(name, 'prob', round(float(p), 4), 'pred_0.5', int(p >= 0.5), 'pred_0.4', int(p >= 0.4), 'pred_0.3', int(p >= 0.3))
