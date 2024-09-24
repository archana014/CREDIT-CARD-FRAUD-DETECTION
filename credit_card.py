# Step 1: Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# Step 2: Load the Dataset
# Replace 'credit_card_data.csv' with the path to your dataset
df = pd.read_csv('/content/creditcard.csv')

# Check class distribution (fraud vs genuine transactions)
print(df['Class'].value_counts())

# Step 3: Data Preprocessing
# Check for missing values
print(df.isnull().sum())

# Feature scaling (assuming 'Amount' is a feature needing scaling)
scaler = StandardScaler()
df['Amount'] = scaler.fit_transform(df[['Amount']])

# Separate features and target
X = df.drop('Class', axis=1)  # Features
y = df['Class']  # Target (0 for genuine, 1 for fraud)

# Step 4: Split the Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train a Classification Model
# You can choose Logistic Regression or Random Forest
# Logistic Regression Model
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)

# Random Forest Model (Alternative)
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)

# Step 6: Evaluate the Model (Without Resampling)
# Predictions
y_pred_lr = lr_model.predict(X_test)
y_pred_rf = rf_model.predict(X_test)

# Evaluation (Logistic Regression)
print("Logistic Regression Metrics:")
print(classification_report(y_test, y_pred_lr))
print(confusion_matrix(y_test, y_pred_lr))

# Evaluation (Random Forest)
print("Random Forest Metrics:")
print(classification_report(y_test, y_pred_rf))
print(confusion_matrix(y_test, y_pred_rf))

# Step 7: Handle Class Imbalance with SMOTE (Oversampling)
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Step 8: Re-train the Model with Resampled Data
# Logistic Regression Model
lr_model_resampled = LogisticRegression()
lr_model_resampled.fit(X_resampled, y_resampled)

# Random Forest Model (Alternative)
rf_model_resampled = RandomForestClassifier()
rf_model_resampled.fit(X_resampled, y_resampled)

# Step 9: Evaluate the Model (With Resampling)
# Predictions
y_pred_lr_resampled = lr_model_resampled.predict(X_test)
y_pred_rf_resampled = rf_model_resampled.predict(X_test)

# Evaluation (Logistic Regression after Resampling)
print("Logistic Regression Metrics (After Resampling):")
print(classification_report(y_test, y_pred_lr_resampled))
print(confusion_matrix(y_test, y_pred_lr_resampled))

# Evaluation (Random Forest after Resampling)
print("Random Forest Metrics (After Resampling):")
print(classification_report(y_test, y_pred_rf_resampled))
print(confusion_matrix(y_test, y_pred_rf_resampled))

# Step 10: Handle Class Imbalance with Random Undersampling (Alternative)
rus = RandomUnderSampler(random_state=42)
X_resampled_undersampled, y_resampled_undersampled = rus.fit_resample(X_train, y_train)

# Train models again with undersampled data
lr_model_undersampled = LogisticRegression()
lr_model_undersampled.fit(X_resampled_undersampled, y_resampled_undersampled)

rf_model_undersampled = RandomForestClassifier()
rf_model_undersampled.fit(X_resampled_undersampled, y_resampled_undersampled)

# Evaluate Logistic Regression after undersampling
y_pred_lr_undersampled = lr_model_undersampled.predict(X_test)
print("Logistic Regression Metrics (After Undersampling):")
print(classification_report(y_test, y_pred_lr_undersampled))

# Evaluate Random Forest after undersampling
y_pred_rf_undersampled = rf_model_undersampled.predict(X_test)
print("Random Forest Metrics (After Undersampling):")
print(classification_report(y_test, y_pred_rf_undersampled))
