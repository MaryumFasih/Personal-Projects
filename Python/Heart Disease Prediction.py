import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve

# Step 1: Load dataset
df = pd.read_csv("C:\\Users\\Maryam Fasih\\Desktop\\internship\\heart_disease_uci.csv")

# Step 2: Data Cleaning
print("Missing values:\n", df.isnull().sum())

# Convert 'num' to binary target (0 = No Disease, 1 = Disease)
df['target'] = (df['num'] > 0).astype(int)

# Drop original 'num' column and rows with missing values
df.drop(columns=['num'], inplace=True)
df.dropna(inplace=True)

# Step 3: Aesthetic EDA Plot
plt.figure(figsize=(8, 6))
ax = sns.countplot(x='target', data=df, palette='Set2')
for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + 0.3, p.get_height() + 3), fontsize=12)
plt.title("Heart Disease Frequency", fontsize=14, weight='bold')
plt.xlabel("Heart Disease (0 = No, 1 = Yes)", fontsize=12)
plt.ylabel("Number of Patients", fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Correlation heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()

# Step 4: Feature Selection
X = df.drop('target', axis=1)
y = df['target']

# Step 5: Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.2, random_state=42)

# Step 6: Standardization
X_mean = X_train.mean(axis=0)
X_std = X_train.std(axis=0)
X_train = (X_train - X_mean) / X_std
X_test = (X_test - X_mean) / X_std

# Step 7: Add bias term
X_train_b = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
X_test_b = np.hstack([np.ones((X_test.shape[0], 1)), X_test])

# Step 8: Sigmoid and training function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def train_logistic_regression(X, y, lr=0.01, epochs=1000):
    m, n = X.shape
    theta = np.zeros((n, 1))
    y = y.reshape(-1, 1)
    for _ in range(epochs):
        z = X @ theta
        h = sigmoid(z)
        gradient = (X.T @ (h - y)) / m
        theta -= lr * gradient
    return theta

theta = train_logistic_regression(X_train_b, y_train, lr=0.1, epochs=10000)

# Step 9: Prediction
def predict(X, theta, threshold=0.5):
    probs = sigmoid(X @ theta)
    return (probs >= threshold).astype(int), probs

y_pred, y_prob = predict(X_test_b, theta)
y_pred = y_pred.flatten()
y_prob = y_prob.flatten()

# Step 10: Evaluation
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print(f"Accuracy: {acc:.2f}")
print("Confusion Matrix:")
print(cm)
print(f"ROC AUC Score: {roc_auc:.2f}")

# Step 11: ROC Curve (aesthetic)
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure(figsize=(8, 5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}", linewidth=2)
plt.plot([0, 1], [0, 1], '--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# Step 12: Feature Importance
feature_importance = pd.Series(theta[1:].flatten(), index=X.columns).sort_values(key=abs, ascending=False)
print("Top features influencing heart disease prediction:")
print(feature_importance.head(10))

plt.figure(figsize=(10, 6))
feature_importance.head(10).plot(kind='barh', color='orange', edgecolor='black')
plt.gca().invert_yaxis()
plt.title("Top 10 Important Features", fontsize=14)
plt.xlabel("Coefficient Magnitude")
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
