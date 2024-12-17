import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    confusion_matrix, 
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path


# Test if in the correct working directory

cwd = Path().resolve()

if not (cwd / "src").is_dir():
    os.chdir(cwd.parent)

# Load the data
def load_data(filepath):
    df = pd.read_csv(filepath)
    
    # Select features (excluding ID and Date)
    features = ['Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio']
    
    X = df[features]
    y = df['Occupancy']
    
    return X, y

# Prepare data
def prepare_data(X, y, test_size=0.2, random_state=42):
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test

# Train Logistic Regression model
def train_logistic_regression(X_train, y_train, max_iter=1000):
    model = LogisticRegression(
        max_iter=max_iter,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

# Evaluate model
def evaluate_model(model, X_test, y_test):
    # Predictions
    y_pred = model.predict(X_test)
    
    # Metrics
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1 Score': f1_score(y_test, y_pred)
    }
    
    return metrics, y_pred

# Visualize Confusion Matrix
def plot_confusion_matrix(y_test, y_pred):
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Not Occupied', 'Occupied'],
                yticklabels=['Not Occupied', 'Occupied'])
    plt.title('Confusion Matrix - Baseline Logistic Regression')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.show()

# Feature Importance
def plot_feature_importance(model, feature_names):
    plt.figure(figsize=(10, 6))
    coefficients = model.coef_[0]
    feature_importance = pd.Series(
        np.abs(coefficients), 
        index=feature_names
    ).sort_values(ascending=False)
    
    sns.barplot(x=feature_importance.values, y=feature_importance.index)
    plt.title('Feature Importance in Logistic Regression')
    plt.xlabel('Importance')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.show()
