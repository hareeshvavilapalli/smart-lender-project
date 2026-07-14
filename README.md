# Smart Lender Project 💳

## Project Overview

Smart Lender is a Machine Learning-based loan approval prediction web application developed using **Flask**. The application predicts whether a loan applicant is likely to receive loan approval based on financial and demographic information such as income, credit history, loan amount, education, and marital status.

The system provides fast and reliable predictions through a clean, responsive, and user-friendly web interface, helping streamline the preliminary loan eligibility screening process.

---

## Project Links 🔗

**Deployment Link:**  
https://acurate.onrender.com/
---
**Demo Video Link:**  
https://drive.google.com/file/d/1IdBb9YjMi8jkgLprV6cKjuKhXx7vvuIp/view?usp=sharing
---
---

## Features 🚀

- Loan Approval Prediction using Machine Learning
- Real-Time Eligibility Prediction
- User-Friendly and Responsive Interface
- Fast and Accurate Predictions
- Flask-Based Backend
- Machine Learning Model Integration

---

## Machine Learning Model 🤖

The prediction model is trained using a loan prediction dataset containing applicant and financial details.

### Features Used for Prediction

- Gender
- Marital Status
- Dependents
- Education
- Self Employment
- Applicant Income
- Co-applicant Income
- Loan Amount
- Loan Amount Term
- Credit History
- Property Area

### Algorithms Considered

- Logistic Regression
- Decision Tree
- Random Forest

The best-performing model was selected and saved as **loan_model.pkl** for prediction.

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

```text
Smart-Lender-project/
│
├── Dataset/
│   └── loan_prediction.csv
│
├── Flask/
│   ├── static/
│   │   ├── style.css
│   │   └── app.js
│   │
│   ├── templates/
│   │   ├── home.html
│   │   ├── index.html
│   │   └── result.html
│   │
│   ├── app.py
│   ├── loan_model.pkl
│   ├── requirements.txt
│   └── Procfile
│
├── README.md
└── .gitignore
```

---

## Installation ⚙

### Clone the Repository

```bash
git clone https://github.com/hareeshvavilapalli/smart-lender-project.git
```

### Navigate to the Project Directory

```bash
cd Smart-Lender-project
```

### Install Required Dependencies

```bash
pip install -r Flask/requirements.txt
```

---

## Run the Application ▶

Navigate to the Flask folder:

```bash
cd Flask
```

Run the application:

```bash
python app.py
```

Open your browser and visit:

```text
http://127.0.0.1:5000
```

---

## Future Enhancements 🔮

- Improve User Interface and User Experience
- Loan EMI Calculator
- Credit Score Analysis
- Secure User Authentication
- Support for Multiple Loan Types
- Cloud Deployment with Continuous Integration
- Enhanced Machine Learning Model Accuracy

---

## Author 👨‍💻

**Hareesh Vavilapalli**

GitHub: https://github.com/hareeshvavilapalli

LinkedIn: https://www.linkedin.com/in/vavilapalli-hareesh-30b308382
