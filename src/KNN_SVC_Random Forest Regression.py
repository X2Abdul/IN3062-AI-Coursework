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
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc
from sklearn.metrics import roc_curve, roc_auc_score


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


# Plot for KNN Before Hyperparameter Tuning
knn_model = train_knn(X_train, y_train)
knn_metrics, knn_preds = evaluate_classification_model(knn_model, X_test, y_test)
print("KNN Metrics:")
print(knn_metrics)
plot_confusion_matrix(y_test, knn_preds, "KNN (Before Tuning)")


# Plot for SVC Before Hyperparameter Tuning
svc_model = train_svc(X_train, y_train)
svc_metrics, svc_preds = evaluate_classification_model(svc_model, X_test, y_test)
print("SVC Metrics:")
print(svc_metrics)
plot_confusion_matrix(y_test, svc_preds, "SVC (Before Tuning)")


# Random Forest Regression
rf_model = train_random_forest(X_train, y_train)
rf_metrics, rf_preds = evaluate_regression_model(rf_model, X_test, y_test)
print("Random Forest Regression Metrics:")
print(rf_metrics)
plot_feature_importance(rf_model, X.columns)


# Plot for Gradient Boosting Before Hyperparameter Tuning
gb_model = train_gradient_boosting(X_train, y_train)
gb_metrics, gb_preds = evaluate_classification_model(gb_model, X_test, y_test)
print("Gradient Boosting Metrics (Before Tuning):")
print(gb_metrics)
plot_confusion_matrix(y_test, gb_preds, "Gradient Boosting (Before Tuning)")






#HYPERPARAMETER TUNINGS FOR MODELS

# Hyperparameter Tuning for KNN
def train_knn_with_tuning(X_train, y_train):
    param_grid = {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance'],
        'p': [1, 2]
    }
    grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='f1')
    grid.fit(X_train, y_train)
    print(f"Best Parameters for KNN: {grid.best_params_}")
    return grid.best_estimator_

# Hyperparameter Tuning for SVC
def train_svc_with_tuning(X_train, y_train):
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    }
    grid = GridSearchCV(SVC(probability=True, random_state=42), param_grid, cv=5, scoring='f1')
    grid.fit(X_train, y_train)
    print(f"Best Parameters for SVC: {grid.best_params_}")
    return grid.best_estimator_


# Hyperparameter Tuning for Gradient Boosting
def train_gradient_boosting_with_tuning(X_train, y_train):
    param_distributions = {
        'n_estimators': randint(50, 200),
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'min_samples_split': randint(2, 10),
        'min_samples_leaf': randint(1, 4),
        'subsample': [0.8, 1.0]
    }
    random_search = RandomizedSearchCV(
        estimator=GradientBoostingClassifier(random_state=42),
        param_distributions=param_distributions,
        n_iter=20,
        cv=3,
        scoring='f1',
        random_state=42
    )
    random_search.fit(X_train, y_train)
    print(f"Best Parameters for Gradient Boosting: {random_search.best_params_}")
    return random_search.best_estimator_




#MODEL PLOTS FOR AFTER HYPERPARAMETER TUNING

# KNN After Tuning
knn_tuned_model = train_knn_with_tuning(X_train, y_train)
knn_tuned_metrics, knn_tuned_preds = evaluate_classification_model(knn_tuned_model, X_test, y_test)
print("KNN After Tuning Metrics:")
print(knn_tuned_metrics)
# plot_confusion_matrix(y_test, knn_tuned_preds, "KNN After Tuning")

# SVC After Tuning
svc_tuned_model = train_svc_with_tuning(X_train, y_train)
svc_tuned_metrics, svc_tuned_preds = evaluate_classification_model(svc_tuned_model, X_test, y_test)
print("SVC After Tuning Metrics:")
print(svc_tuned_metrics)
# plot_confusion_matrix(y_test, svc_tuned_preds, "SVC After Tuning")

# Gradient Boosting After Tuning
gb_tuned_model = train_gradient_boosting_with_tuning(X_train, y_train)
gb_tuned_metrics, gb_tuned_preds = evaluate_classification_model(gb_tuned_model, X_test, y_test)
print("Gradient Boosting Metrics (After Tuning):")
print(gb_tuned_metrics)
plot_confusion_matrix(y_test, gb_tuned_preds, "Gradient Boosting (After Tuning)")





#MODEL PLOTS FOR AFTER HYPERPARAMETER TUNING

# KNN After Tuning
knn_tuned_model = train_knn_with_tuning(X_train, y_train)
knn_tuned_metrics, knn_tuned_preds = evaluate_classification_model(knn_tuned_model, X_test, y_test)
print("KNN After Tuning Metrics:")
print(knn_tuned_metrics)
# plot_confusion_matrix(y_test, knn_tuned_preds, "KNN After Tuning")

# SVC After Tuning
svc_tuned_model = train_svc_with_tuning(X_train, y_train)
svc_tuned_metrics, svc_tuned_preds = evaluate_classification_model(svc_tuned_model, X_test, y_test)
print("SVC After Tuning Metrics:")
print(svc_tuned_metrics)
# plot_confusion_matrix(y_test, svc_tuned_preds, "SVC After Tuning")

# Gradient Boosting After Tuning
gb_tuned_model = train_gradient_boosting_with_tuning(X_train, y_train)
gb_tuned_metrics, gb_tuned_preds = evaluate_classification_model(gb_tuned_model, X_test, y_test)
print("Gradient Boosting Metrics (After Tuning):")
print(gb_tuned_metrics)
plot_confusion_matrix(y_test, gb_tuned_preds, "Gradient Boosting (After Tuning)")






# PRECISION AND RECALL PLOT FOR AFTER HYPERPARAMETER TUNING MODELS

def plot_precision_recall_comparison(models, X_test, y_test):
    
    plt.figure(figsize=(10, 7))
    
    for name, model in models.items():
        # Get predicted probabilities (for models that support predict_proba)
        if hasattr(model, "predict_proba"):
            y_scores = model.predict_proba(X_test)[:, 1]
        else:  # For models like SVM that use decision_function
            y_scores = model.decision_function(X_test)
        
        # Calculate precision-recall
        precision, recall, _ = precision_recall_curve(y_test, y_scores)
        auc_pr = auc(recall, precision)
        
        # Plot curve
        plt.plot(recall, precision, label=f'{name} (AUC = {auc_pr:.2f})')
    
    plt.title('Precision-Recall Curve (After Tuning)', fontsize=16)
    plt.xlabel('Recall', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    plt.legend(loc='lower left', fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

tuned_models = {
    "KNN": knn_tuned_model,  # Trained KNN model after tuning
    "SVC": svc_tuned_model,  # Trained SVC model after tuning
    "Gradient Boosting": gb_tuned_model, # Trained Gradient Boosting model after tuning
}

# Call the function with trained models, X_test, and y_test
plot_precision_recall_comparison(tuned_models, X_test, y_test)







# ROC CURVE PLOT FOR AFTER HYPERPARAMETER TUNING MODELS


# Function to plot ROC Curve
def plot_roc_curve(models, X_test, y_test, title="ROC Curve for Tuned Models"):
    plt.figure(figsize=(10, 8))
    for model_name, model in models.items():
        y_prob = model.predict_proba(X_test)[:, 1]  # Get probability for class 1
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc:.2f})")
    
    plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line for random guessing
    plt.title(title)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.grid()
    plt.tight_layout()
    plt.show()

# Dictionary of tuned models
tuned_models = {
    "KNN After Tuning": knn_tuned_model,
    "SVC After Tuning": svc_tuned_model,
    "Gradient Boosting": gb_tuned_model,
}

# Plot ROC curve
plot_roc_curve(tuned_models, X_test, y_test)


