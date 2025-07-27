# predict.py
#to see if task 2 works proprerly
import joblib
import pandas as pd

# -------------------------------
# 1. Load your trained pipeline
# -------------------------------
# Change this to 'telco_rf_pipeline.joblib' if you prefer Random Forest
pipeline = joblib.load('telco_logreg_pipeline.joblib')

print("✅ Pipeline loaded successfully!")

# -------------------------------
# 2. Create a new customer record
# -------------------------------
# This must match your training feature columns!
new_customer = pd.DataFrame([{
    'gender': 'Female',
    'SeniorCitizen': 0,
    'Partner': 'Yes',
    'Dependents': 'No',
    'tenure': 24,
    'PhoneService': 'Yes',
    'MultipleLines': 'No',
    'InternetService': 'Fiber optic',
    'OnlineSecurity': 'No',
    'OnlineBackup': 'Yes',
    'DeviceProtection': 'Yes',
    'TechSupport': 'No',
    'StreamingTV': 'Yes',
    'StreamingMovies': 'Yes',
    'Contract': 'Month-to-month',
    'PaperlessBilling': 'Yes',
    'PaymentMethod': 'Electronic check',
    'MonthlyCharges': 80.5,
    'TotalCharges': 2000.0
}])

# -------------------------------
# 3. Make prediction
# -------------------------------
prediction = pipeline.predict(new_customer)[0]
probability = pipeline.predict_proba(new_customer)[0][1]  # Probability of churn

print("\n=== Prediction ===")
print("Predicted Churn:", "Yes" if prediction == 1 else "No")
print(f"Probability of Churn: {probability:.2%}")

# -------------------------------
# ✅ If you see 'Yes' or 'No' with probability,
# ✅ your pipeline works perfectly!
# -------------------------------
