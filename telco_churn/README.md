
# Telco Customer Churn Prediction Pipeline

This project implements a machine learning pipeline to predict customer churn for a telecommunications company using the Telco Customer Churn dataset. It employs scikit-learn to build and evaluate two models: Logistic Regression and Random Forest, with preprocessing and hyperparameter tuning via GridSearchCV. Additionally, a prediction script (`predict.py`) is provided to demonstrate how to use the trained models for new customer predictions.

## Table of Contents

* Overview
* Features
* Requirements
* Setup
* Data Preparation
* Pipeline Architecture
* Training and Hyperparameter Tuning
* Evaluation
* Prediction with predict.py
* Usage
* File Structure
* Contributing
* License

## Overview

The `telco_churn_pipeline.py` script processes the Telco Customer Churn dataset to predict whether a customer will churn (Yes or No). It uses a preprocessing pipeline to handle numeric and categorical features, followed by training and evaluating Logistic Regression and Random Forest models with hyperparameter tuning. The `predict.py` script loads a trained pipeline to make predictions on new customer data.

## Features

* Preprocessing Pipeline: Handles numeric (scaling) and categorical (one-hot encoding) features using ColumnTransformer.
* Model Training: Implements Logistic Regression and Random Forest classifiers.
* Hyperparameter Tuning: Uses GridSearchCV to optimize model parameters.
* Evaluation: Provides classification reports with precision, recall, and F1-score.
* Model Persistence: Saves trained pipelines for later use with joblib.
* Prediction: Includes `predict.py` for making churn predictions on new customer records.

## Requirements

* Python 3.8+
* pandas
* scikit-learn
* joblib

Install dependencies using:

```
pip install pandas scikit-learn joblib
```

## Setup

**Clone the Repository:**

```
git clone <repository-url>
cd <repository-directory>
```

**Download the Dataset:**

Obtain the `WA_Fn-UseC_-Telco-Customer-Churn.csv` dataset (e.g., from Kaggle) and place it in the project directory.
The dataset should include columns like `customerID`, `Churn`, `TotalCharges`, and other features.

## Data Preparation

* **Dataset**: The script loads `WA_Fn-UseC_-Telco-Customer-Churn.csv` using pandas.
* **Cleaning**:

  * Strips whitespace from column names.
  * Converts `TotalCharges` to numeric, dropping rows with missing values.
* **Features and Target**:

  * Features (X): All columns except `customerID` and `Churn`.
  * Target (y): `Churn` column, mapped to binary values (Yes → 1, No → 0).
* **Train/Test Split**: Splits data into 80% training and 20% testing sets with stratification.

## Pipeline Architecture

* **Preprocessing**:

  * Numeric Features: Scaled using `StandardScaler`.
  * Categorical Features: One-hot encoded using `OneHotEncoder` (handles unknown categories).
  * Uses `ColumnTransformer` to apply transformations to respective feature types.
* **Models**:

  * Logistic Regression: Configured with `max_iter=1000`.
  * Random Forest: Configured with `random_state=42`.
* **Hyperparameter Tuning**:

  * Logistic Regression: Tunes `C` (inverse regularization strength) and `solver`.
  * Random Forest: Tunes `n_estimators`, `max_depth`, and `min_samples_split`.

## Training and Hyperparameter Tuning

* **GridSearchCV**:

  * Performs 5-fold cross-validation to find the best hyperparameters based on accuracy.
  * Reports best parameters and cross-validation score for each model.
* **Training**: Fits both pipelines on the training data.

## Evaluation

Evaluates both models on the test set using `classification_report`, reporting:

* Precision, recall, F1-score, and support for each class.
* Macro and weighted averages for overall performance.

## Prediction with predict.py

The `predict.py` script demonstrates how to use a trained pipeline to predict churn for a new customer. It:

* Loads a saved pipeline (`telco_logreg_pipeline.joblib` or `telco_rf_pipeline.joblib`).
* Creates a new customer record as a pandas DataFrame with features matching the training data (e.g., `gender`, `tenure`, `MonthlyCharges`, etc.).
* Outputs the predicted churn (Yes or No) and the probability of churn.

**Example Usage in predict.py:**

```python
import joblib
import pandas as pd

pipeline = joblib.load('telco_logreg_pipeline.joblib')
new_customer = pd.DataFrame([{
    'gender': 'Female',
    'SeniorCitizen': 0,
    # ... other features ...
    'MonthlyCharges': 80.5,
    'TotalCharges': 2000.0
}])
prediction = pipeline.predict(new_customer)[0]
probability = pipeline.predict_proba(new_customer)[0][1]
print("Predicted Churn:", "Yes" if prediction == 1 else "No")
print(f"Probability of Churn: {probability:.2%}")
```

## Usage

**Training:**

1. Ensure `WA_Fn-UseC_-Telco-Customer-Churn.csv` is in the project directory.
2. Run the training script:

```
python telco_churn_pipeline.py
```

The script will:

* Load and preprocess the dataset.
* Train and tune Logistic Regression and Random Forest models.
* Print best parameters, cross-validation scores, and classification reports.
* Save trained pipelines as `telco_logreg_pipeline.joblib` and `telco_rf_pipeline.joblib`.

**Prediction:**

1. Ensure the saved pipeline files are in the project directory.
2. Run the prediction script:

```
python predict.py
```

Modify `predict.py` to use either `telco_logreg_pipeline.joblib` or `telco_rf_pipeline.joblib` and adjust the `new_customer` DataFrame as needed.
The script will output the predicted churn and probability.

## File Structure

```
project_directory/
├── WA_Fn-UseC_-Telco-Customer-Churn.csv  # Input dataset
├── telco_churn_pipeline.py               # Main training script
├── predict.py                            # Prediction script
├── telco_logreg_pipeline.joblib          # Saved Logistic Regression pipeline
├── telco_rf_pipeline.joblib              # Saved Random Forest pipeline
└── README.md                             # This file
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature-branch`).
3. Commit changes (`git commit -m "Add feature"`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a pull request.


