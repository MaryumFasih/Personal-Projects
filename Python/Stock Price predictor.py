import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# 1) Fetching data
ticker = 'AAPL'
data = yf.download(ticker, start="2022-01-01", end="2024-12-31")

# 2) Data preprocessing
data = data[['Open', 'High', 'Low', 'Volume', 'Close']]
data.dropna(inplace=True)

data['Target'] = data['Close'].shift(-1)
data.dropna(inplace=True)

X = data[['Open', 'High', 'Low', 'Volume']].values
y = data['Target'].values.reshape(-1, 1)

# 3) Split data into training and testing 
split_index = int(len(X) * 0.8)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# 4) Add bias
X_train_b = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
X_test_b = np.hstack([np.ones((X_test.shape[0], 1)), X_test])

# 5)Linear Regression using Normal Equation
theta = np.linalg.inv(X_train_b.T.dot(X_train_b)).dot(X_train_b.T).dot(y_train)

# 6) Predictions
predictions = X_test_b.dot(theta)

# 7) Plotting
plt.figure(figsize=(12, 6))
plt.plot(y_test.flatten(), label='Actual Close Price', linewidth=2)
plt.plot(predictions.flatten(), label='Predicted Close Price', linewidth=2)
plt.title(f"{ticker} - Actual vs Predicted Closing Price (Manual Linear Regression)")
plt.xlabel('Days')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 8) Printing
mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error: {mse:.2f}")
