import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    confusion_matrix,
    mean_squared_error, 
    r2_score
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestRegressor
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

# Train KNN
def train_knn(X_train, y_train, n_neighbors=5, weights='uniform', p=2):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, p=p)
    knn.fit(X_train, y_train)
    return knn

# Train SVC
def train_svc(X_train, y_train, kernel='linear'):
    svc = SVC(kernel=kernel, probability=True, random_state=42)
    svc.fit(X_train, y_train)
    return svc

# Train Random Forest (regression)
def train_random_forest(X_train, y_train, n_estimators=100):
    rf = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
    rf.fit(X_train, y_train)
    return rf

# Evaluate classification model
def evaluate_classification_model(model, X_test, y_test):
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

# Evaluate regression model
def evaluate_regression_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return {'MSE': mse, 'R2': r2}, y_pred

# Plot confusion matrix
def plot_confusion_matrix(y_test, y_pred, title):
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Not Occupied', 'Occupied'],
                yticklabels=['Not Occupied', 'Occupied'])
    plt.title(f'Confusion Matrix - {title}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.show()

# Plot feature importance
def plot_feature_importance(model, feature_names):
    plt.figure(figsize=(10, 6))
    importances = model.feature_importances_
    feature_importance = pd.Series(importances, index=feature_names).sort_values(ascending=False)
    sns.barplot(x=feature_importance.values, y=feature_importance.index)
    plt.title('Feature Importance in Random Forest')
    plt.xlabel('Importance')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.show()


# Load dataset
X, y = load_data('datasets/usable/datatraining.txt')

# Prepare data
X_train, X_test, y_train, y_test = prepare_data(X, y)


# KNN
knn_model = train_knn(X_train, y_train)
knn_metrics, knn_preds = evaluate_classification_model(knn_model, X_test, y_test)
print("KNN Metrics:")
print(knn_metrics)
plot_confusion_matrix(y_test, knn_preds, "KNN")

# SVC
svc_model = train_svc(X_train, y_train)
svc_metrics, svc_preds = evaluate_classification_model(svc_model, X_test, y_test)
print("SVC Metrics:")
print(svc_metrics)
plot_confusion_matrix(y_test, svc_preds, "SVC")

# Random Forest Regression
rf_model = train_random_forest(X_train, y_train)
rf_metrics, rf_preds = evaluate_regression_model(rf_model, X_test, y_test)
print("Random Forest Regression Metrics:")
print(rf_metrics)
plot_feature_importance(rf_model, X.columns)

