from fastapi import FastAPI
import joblib
import numpy as np

# Load model
model = joblib.load("models/model.pkl")

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Customer Churn Prediction API is running"}

@app.post("/predict")
def predict(data: dict):

    # Convert input into model format
    input_data = np.array([[
        data["tenure"],
        data["monthly_charges"],
        data["total_charges"],
        data["support_tickets"],
        data["contract"],
        data["payment_delay"]
    ]])

    # Prediction
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    return {
        "churn_prediction": int(prediction),
        "churn_probability": float(probability)
    }