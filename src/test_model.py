import joblib
import pandas as pd

model = joblib.load("models/model.pkl")

# use SAME feature names as training
sample = pd.DataFrame([{
    "tenure": 12,
    "monthly_charges": 70,
    "total_charges": 2000,
    "support_tickets": 5,
    "contract": 0,
    "payment_delay": 15
}])

prediction = model.predict(sample)[0]
probability = model.predict_proba(sample)[0][1]

print("Churn Prediction:", prediction)
print("Churn Probability:", probability)