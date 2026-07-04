# Smart Lender Project 💳

## Project Overview
Smart Lender is a Machine Learning-based loan approval prediction web application built using Flask. The application helps determine whether a loan applicant is likely to get loan approval based on financial and demographic details such as income, credit history, loan amount, education, and marital status.

This system provides fast predictions through a clean and user-friendly interface, helping streamline loan eligibility screening.

---

## Features 🚀

- Loan Approval Prediction using Machine Learning
- Real-Time Eligibility Check
- User-Friendly Web Interface
- Fast and Accurate Predictions
- Flask-Based Backend Integration
- Responsive Design for Better User Experience

---

## Machine Learning Model 🤖

The model is trained using a loan prediction dataset containing applicant and financial details.

### Features Used for Prediction:
- Gender
- Marital Status
- Dependents
- Education
- Self Employment
- Applicant Income
- Coapplicant Income
- Loan Amount
- Loan Amount Term
- Credit History
- Property Area

### Algorithms Considered:
- Logistic Regression  
- Decision Tree  
- Random Forest  

Selected model provides reliable loan prediction results based on historical data.

---

## Tech Stack 🛠

### Backend
- Python
- Flask

### Frontend
- HTML
- CSS
- JavaScript

### Machine Learning
- Scikit-learn
- Pandas
- NumPy

---

## Project Structure 📂

```bash
Smart-Lender-project/
│
├── Dataset/
│   └── loan_prediction.csv
│
├── Flask/
│   ├── static/
│   ├── templates/
│   ├── app.py
│   ├── loan_model.pkl
│   ├── requirements.txt
│   └── Procfile
```

---

## Installation ⚙

### Clone Repository

```bash
git clone https://github.com/hareeshvavilapalli/smart-lender-project.git
```

### Navigate to Project Directory

```bash
cd Smart-Lender-project
```

### Install Dependencies

```bash
pip install -r Flask/requirements.txt
```

---

## Run Application ▶

```bash
cd Flask
python app.py
```

Open browser and visit:

```bash
http://127.0.0.1:5000
```

---

## Future Enhancements 🔮

- Improve UI/UX
- Loan EMI Calculator
- Credit Score Analysis
- Cloud Deployment
- Authentication System

---

## Author 👨‍💻

**Hareesh Vavilapalli**

GitHub: [hareeshvavilapalli](https://github.com/hareeshvavilapalli)
