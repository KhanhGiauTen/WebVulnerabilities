import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time
import seaborn as sns
from sklearn.metrics import roc_curve

from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, log_loss

def load_data():
    np.load('data/X_train.npy')
    np.load('data/y_train.npy')
    np.load('data/X_test.npy')
    np.load('data/y_test.npy')
    np.load('data/X_val.npy')
    np.load('data/y_val.npy')
    
# Function to calculate and print indices
def print_metrics(train_type,y_true, y_pred, y_proba, dataset_name):
    accuracy = accuracy_score(y_true, y_pred)
    test_error = 1 - accuracy
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_proba)
    logloss = log_loss(y_true, y_proba)
    
    print(f"\nValidation on {dataset_name}:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Test Error (1-Accuracy): {test_error:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"Log Loss: {logloss:.4f}")  
    
    
    return {
        'Train Type': train_type,
        'Dataset': dataset_name,
        'Accuracy': accuracy,
        'Test Error': test_error,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'ROC-AUC': roc_auc,
        'Log Loss': logloss        
    }
    
    
def random_forest(X_train, y_train, X_test, y_test, X_val, y_val):
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    
    #Training
    start_time = time.time()
    rf_model.fit(X_train, y_train)
    training_time = time.time() - start_time
    print(f"Training time: {training_time:.2f} seconds")
    
    #Prediction
    start_time = time.time()
    y_val_pred = rf_model.predict(X_val)
    y_val_proba = rf_model.predict_proba(X_val)[:, 1]
    y_test_pred = rf_model.predict(X_test)
    y_test_proba = rf_model.predict_proba(X_test)[:, 1]
    prediction_time = time.time() - start_time
    print(f"Prediction time(s): {prediction_time:.2f}")
    
    
    # Calculate indices
    val_metrics = print_metrics('ML',y_val, y_val_pred, y_val_proba,"tập val")
    test_metrics = print_metrics('ML',y_test, y_test_pred, y_test_proba, "tập test")
    test_metrics['Training Time'] = training_time
    test_metrics['Prediction Time'] = prediction_time
    
    
def random_forest_grid_search(X_train, y_train, X_test, y_test, X_val, y_val):
    # Define hyperparameter 
    param_grid = {
        'n_estimators': [50, 100, 200],  
        'max_depth': [10, 20, None],   
        'min_samples_split': [2, 5]      
    }

    # Initial Grid Search
    rf_model = RandomForestClassifier(random_state=42, n_jobs=-1)
    grid_search = GridSearchCV(
        estimator=rf_model,
        param_grid=param_grid,
        cv=5, 
        scoring='recall', 
        n_jobs=-1,  
        verbose=1
    )

    # Training
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    training_time = time.time() - start_time
    print(f"Tranining time (s): {training_time:.2f}")


    # Get best model
    best_rf_model = grid_search.best_estimator_
    print("\nBest hyperparameter :", grid_search.best_params_)
    print("Best recall on cross-validation:", grid_search.best_score_)

    # Prediction
    start_time = time.time()
    y_val_pred = best_rf_model.predict(X_val)
    y_val_proba = best_rf_model.predict_proba(X_val)[:, 1]
    y_test_pred = best_rf_model.predict(X_test)
    y_test_proba = best_rf_model.predict_proba(X_test)[:, 1]
    prediction_time = time.time() - start_time
    print(f"Prediction time (s): {prediction_time:.2f}")

    # Calculate indices
    val_metrics = print_metrics('ML',y_val, y_val_pred, y_val_proba,"tập val")
    test_metrics = print_metrics('ML',y_test, y_test_pred, y_test_proba, "tập test")
    test_metrics['Training Time'] = training_time
    test_metrics['Prediction Time'] = prediction_time

    # Save indices
    results_df = pd.DataFrame([val_metrics, test_metrics])
    results_df['Model'] = 'Random Forest Optimized'
    try:
        existing_results = pd.read_csv('model_results.csv')
        results_df = pd.concat([existing_results, results_df], ignore_index=True)
    except FileNotFoundError:
        pass
    results_df.to_csv('model_results.csv', index=False)
    print("\nThe results was saved in 'model_results.csv'")
    


    
    
    