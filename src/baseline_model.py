import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    confusion_matrix, 
    classification_report
)

# --------------------------------------------------------------------------------------------


def load_data(filepath):
    df = pd.read_csv(filepath)
    
    # Select features (excluding ID and Date)

    features = ['Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio']
    
    X = df[features]
    y = df['Occupancy']
    
    return X, y


def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


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


def print_metrics(dataset_name, metrics):

    print(f"{dataset_name} results:")

    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")


def plot_confusion_matrix(dataset_name, y_test, y_pred):

    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Not Occupied', 'Occupied'],
                yticklabels=['Not Occupied', 'Occupied'])
    
    plt.title(f'{dataset_name} Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.show()


def plot_feature_importance(dataset_name, feature_names):

    plt.figure(figsize=(10, 6))
    coefficients = model.coef_[0]

    feature_importance = pd.Series(
        np.abs(coefficients), 
        index=feature_names
    ).sort_values(ascending=False)
    
    sns.barplot(x=feature_importance.values, y=feature_importance.index)
    plt.title(f'{dataset_name} Feature Importance')
    plt.xlabel('Importance')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.show()


# --------------------------------------------------------------------------------------------


# Test if in the correct working directory, else change current working directory

cwd = Path().resolve()

if not (cwd / "src").is_dir():
    os.chdir(cwd.parent)

# Move to the datasets directory and make the usable directory

os.chdir(cwd / "datasets/usable")
cwd = Path().resolve()

# Name the datasets

training = "datatraining.txt"
testing1 = "datatest1.txt"
testing2 = "datatest2.txt"

# Load the training dataset

X_train, y_train = load_data(cwd / training)

# Load the testing datasets

X_test1, y_test1 = load_data(cwd / testing1)
X_test2, y_test2 = load_data(cwd / testing2)

# Train the model

model = train_logistic_regression(X_train, y_train)

# Evaluate model

metrics1, y_pred1 = evaluate_model(model, X_test1, y_test1)
metrics2, y_pred2 = evaluate_model(model, X_test2, y_test2)

"""
# Prepare the data
X_train, X_test, y_train, y_test = prepare_data(X, y)

# Train the model
model = train_logistic_regression(X_train, y_train)

# Evaluate model
metrics, y_pred = evaluate_model(model, X_test, y_test)
"""

# Print test metrics

print("-----------------------------------------------------")

print_metrics(testing1, metrics1)

print("-----------------------------------------------------")

print_metrics(testing2, metrics2)

print("-----------------------------------------------------")

# Print detailed classification report

print(classification_report(y_test1, y_pred1))

print("-----------------------------------------------------")

print(classification_report(y_test2, y_pred2))

print("-----------------------------------------------------")

# Visualizations

plot_confusion_matrix(testing1, y_test1, y_pred1)
plot_feature_importance(testing1, X_test1.columns)

plot_confusion_matrix(testing2, y_test2, y_pred2)
plot_feature_importance(testing2, X_test2.columns)