# telco_churn_pipeline.py

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# ---------------------------
# 1. Load dataset
# ---------------------------
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Remove any leading/trailing spaces in column names
df.columns = df.columns.str.strip()

# Example: If 'TotalCharges' has missing values, fill or drop them
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna(subset=['TotalCharges'])

# ---------------------------
# 2. Features and Target
# ---------------------------
X = df.drop(['customerID', 'Churn'], axis=1)
y = df['Churn'].map({'Yes': 1, 'No': 0})

# ---------------------------
# 3. Train/Test split
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------------
# 4. Preprocessing pipeline
# ---------------------------
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# ---------------------------
# 5. Logistic Regression Pipeline & GridSearchCV
# ---------------------------
logreg_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])

logreg_param_grid = {
    'classifier__C': [0.1, 1.0, 10.0],
    'classifier__solver': ['liblinear', 'lbfgs']
}

logreg_grid = GridSearchCV(
    logreg_pipeline, logreg_param_grid,
    cv=5, scoring='accuracy'
)

logreg_grid.fit(X_train, y_train)

print("\n=== Logistic Regression ===")
print("Best params:", logreg_grid.best_params_)
print("Best CV score:", logreg_grid.best_score_)

y_pred_logreg = logreg_grid.predict(X_test)
print("\nLogistic Regression Classification Report:")
print(classification_report(y_test, y_pred_logreg))

# Save Logistic Regression Pipeline
joblib.dump(logreg_grid.best_estimator_, 'telco_logreg_pipeline.joblib')
print("Logistic Regression Pipeline saved as 'telco_logreg_pipeline.joblib'")

# ---------------------------
# 6. Random Forest Pipeline & GridSearchCV
# ---------------------------
rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

rf_param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5],
}

rf_grid = GridSearchCV(
    rf_pipeline, rf_param_grid,
    cv=5, scoring='accuracy'
)

rf_grid.fit(X_train, y_train)

print("\n=== Random Forest ===")
print("Best params:", rf_grid.best_params_)
print("Best CV score:", rf_grid.best_score_)

y_pred_rf = rf_grid.predict(X_test)
print("\nRandom Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))

# Save Random Forest Pipeline
joblib.dump(rf_grid.best_estimator_, 'telco_rf_pipeline.joblib')
print("Random Forest Pipeline saved as 'telco_rf_pipeline.joblib'")

# ---------------------------
# 7. How to load later:
# ---------------------------
print("\nTo load your pipelines later:")
print("  import joblib")
print("  model = joblib.load('telco_logreg_pipeline.joblib')  # or 'telco_rf_pipeline.joblib'")
print("  predictions = model.predict(new_data)")
