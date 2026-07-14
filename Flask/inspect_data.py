import pandas as pd
from pathlib import Path
import pickle

csv_path = Path('../Dataset/loan_prediction.csv')
df = pd.read_csv(csv_path)
print('Columns:', list(df.columns))
print('\nTarget counts:')
print(df['Loan_Status'].value_counts(dropna=False))
print('\nUnique values:')
for col in ['Gender','Married','Education','Self_Employed','Dependents','Property_Area','Credit_History','Loan_Status']:
    print(col, df[col].unique())
print('\nRows:', len(df))
model_path = Path('loan_model.pkl')
print('Model exists', model_path.exists())
if model_path.exists():
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print('Loaded object type:', type(model))
    if hasattr(model, 'classes_'):
        print('Classes:', model.classes_)
