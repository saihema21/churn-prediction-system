import pandas as pd
import numpy as np

np.random.seed(42)

n = 5000  # number of customers

df = pd.DataFrame({
    "tenure": np.random.randint(1, 72, n),
    "monthly_charges": np.random.randint(20, 120, n),
    "total_charges": np.random.randint(100, 8000, n),
    "support_tickets": np.random.randint(0, 10, n),
    "contract": np.random.choice([0, 1, 2], n),  # 0 = month-to-month
    "payment_delay": np.random.randint(0, 30, n),
})

# REALISTIC churn logic (important)
df["churn"] = (
    (df["contract"] == 0) &
    (df["support_tickets"] > 3) &
    (df["payment_delay"] > 10)
).astype(int)

# Save dataset
df.to_csv("data/churn_data.csv", index=False)

print("✅ Dataset created successfully!")
print("Shape:", df.shape)
print(df["churn"].value_counts())