import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Step 1: Load dataset
df = pd.read_csv("C:\\Users\\Maryam Fasih\\Desktop\\internship\\kc_house_data.csv")  # Download this from Kaggle

# Step 2: Clean and inspect
df.drop(columns=['id', 'date'], inplace=True)
print("Missing values:\n", df.isnull().sum())

# Step 3: Select relevant features
features = ['sqft_living', 'bedrooms', 'bathrooms', 'floors', 'zipcode']
X = df[features]
y = df['price'].values.reshape(-1, 1)

# One-hot encode location
X = pd.get_dummies(X, columns=['zipcode'], drop_first=True)

# Step 4: Train/test split
X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size=0.2, random_state=42)

# Step 5: Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 6: Add bias term manually
X_train_b = np.hstack([np.ones((X_train_scaled.shape[0], 1)), X_train_scaled])
X_test_b = np.hstack([np.ones((X_test_scaled.shape[0], 1)), X_test_scaled])

# Step 7: Manually implement Linear Regression using Normal Equation
def train_linear_regression(X, y):
    theta = np.linalg.inv(X.T @ X) @ X.T @ y
    return theta

theta = train_linear_regression(X_train_b, y_train)

# Step 8: Predict
y_pred = X_test_b @ theta

# Step 9: Evaluation
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"Mean Absolute Error (MAE): ${mae:,.2f}")
print(f"Root Mean Squared Error (RMSE): ${rmse:,.2f}")

# Step 10: Plot predictions vs actual
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.4, color='purple')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Prices (Manual Linear Regression)")
plt.grid(True)
plt.tight_layout()
plt.show()
