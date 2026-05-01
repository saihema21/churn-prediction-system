# 🚀 Customer Churn Prediction System

## 📌 Project Overview
This project is a Machine Learning-based Customer Churn Prediction System that predicts whether a customer is likely to churn based on behavioral, billing, and engagement data. The goal is to help businesses reduce customer loss and improve retention strategies using data-driven insights.

---

## 🎯 Objective
To build an end-to-end machine learning pipeline that:
- Predicts customer churn (Yes/No)
- Uses customer behavior and billing data
- Provides a trained ML model for inference
- Enables quick testing through a Python script

---

## 📊 Problem Statement
Customer churn is a major challenge for subscription-based businesses (Telecom, SaaS, OTT, Fintech). This project helps identify at-risk customers early so that companies can take preventive actions like discounts, support improvements, or engagement campaigns.

---

## 🧠 Machine Learning Approach

### Dataset Features:
- Tenure
- Monthly Charges
- Total Charges
- Support Tickets
- Contract Type
- Payment Delay

### Target Variable:
- `churn` (0 = No, 1 = Yes)

### Model Used:
- Random Forest Classifier

---

## ⚙️ Tech Stack
- Python 🐍
- Pandas
- NumPy
- Scikit-learn
- Joblib

---

## 📁 Project Structure
Customer-Churn-Prediction/
│
├── data/ # Dataset
├── src/ # Source code
│ ├── create_data.py
│ ├── train.py
│ └── test_model.py
├── models/ # Trained model
│ └── model.pkl
├── requirements.txt
└── README.md


---

📊 Output Example
Churn Prediction: 1
Churn Probability: 0.82


## 🚀 How to Run This Project

### 1️⃣ Install dependencies
```bash
pip install -r requirements.txt
2️⃣ Generate dataset
python src/create_data.py
3️⃣ Train model
python src/train.py
4️⃣ Test model
python src/test_model.py.

