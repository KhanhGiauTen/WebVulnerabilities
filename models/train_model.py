import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import time
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns
from sklearn.metrics import roc_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, log_loss
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import GaussianNB
from preprocessing.preprocessing import CSIC_preprocess, Malicious_phish_preprocess
from xgboost import XGBClassifier
 
    
def random_forest(X_train, y_train, X_test, X_val):
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=2)
    
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
    
    
    return y_test_pred, y_test_proba, y_val_pred, y_val_proba
    
    
def random_forest_grid_search(X_train, y_train, X_test, y_test, X_val, y_val):
    # Define hyperparameter 
    param_grid = {
        'n_estimators': [50, 100],  
        'max_depth': [10, None],   
        'min_samples_split': [2,5]      
    }

    # Initial Grid Search
    rf_model = RandomForestClassifier(random_state=42, n_jobs=2)
    grid_search = GridSearchCV(
        estimator=rf_model,
        param_grid=param_grid,
        cv=3, 
        scoring='recall', 
        n_jobs=2,  
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

    
    return y_test_pred, y_test_proba, y_val_pred, y_val_proba, best_rf_model
    

 
def naive_bayes_opt_gs(X_train, y_train, X_test, y_test, X_val, y_val):
    nb_model = GaussianNB()
    start_time = time.time()
    nb_model.fit(X_train, y_train)  
    training_time = time.time() - start_time
    print(f"Training time (s):{training_time:.2f}")

    # Prediction
    start_time = time.time()
    y_val_pred = nb_model.predict(X_val)
    y_val_proba = nb_model.predict_proba(X_val)[:, 1]
    y_test_pred = nb_model.predict(X_test)
    y_test_proba = nb_model.predict_proba(X_test)[:, 1]
    prediction_time = time.time() - start_time
    print(f"Prediction time (s):{prediction_time:.2f}")
    return y_test_pred, y_test_proba, y_val_pred, y_val_proba,nb_model
    
def decision_tree(X_train, y_train, X_test, y_test, X_val, y_val):
    dt_model = DecisionTreeClassifier(random_state=42)

    start_time = time.time()
    dt_model.fit(X_train, y_train)
    training_time = time.time() - start_time
    print(f"Training time (s):{training_time:.2f}")

    # Prediction
    start_time = time.time()
    y_val_pred = dt_model.predict(X_val)
    y_val_proba = dt_model.predict_proba(X_val)[:, 1]
    y_test_pred = dt_model.predict(X_test)
    y_test_proba = dt_model.predict_proba(X_test)[:, 1]
    prediction_time = time.time() - start_time
    print(f"Prediction time (s):{prediction_time:.2f}")
    return y_test_pred, y_test_proba, y_val_pred, y_val_proba,dt_model

def knn(X_train, y_train, X_test, y_test, X_val, y_val):
    knn_model = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    start_time = time.time()
    knn_model.fit(X_train, y_train)
    training_time = time.time() - start_time
    print(f"Training time (s):{training_time:.2f}")

    # Prediction
    start_time = time.time()
    y_val_pred = knn_model.predict(X_val)
    y_val_proba = knn_model.predict_proba(X_val)[:, 1]
    y_test_pred = knn_model.predict(X_test)
    y_test_proba = knn_model.predict_proba(X_test)[:, 1]
    prediction_time = time.time() - start_time
    print(f"Prediction time (s):{prediction_time:.2f}")
    return y_test_pred, y_test_proba, y_val_pred, y_val_proba,knn_model

def linear_svc(X_train, y_train, X_test, y_test, X_val, y_val):
    base_model = LinearSVC(C=1.0, random_state=42, tol=0.01, max_iter=1000)
    svm_model = CalibratedClassifierCV(base_model, method='sigmoid', cv=5)

    # Training
    start_time = time.time()
    svm_model.fit(X_train, y_train)
    training_time = time.time() - start_time
    print(f"Training time (s):{training_time:.2f}")

    # Prediction
    start_time = time.time()
    y_val_pred = svm_model.predict(X_val)
    y_val_proba = svm_model.predict_proba(X_val)[:, 1]
    y_test_pred = svm_model.predict(X_test)
    y_test_proba = svm_model.predict_proba(X_test)[:, 1]
    prediction_time = time.time() - start_time
    print(f"Prediction time (s):{prediction_time:.2f}")
    return y_test_pred, y_test_proba, y_val_pred, y_val_proba,svm_model

def xgboost(X_train, y_train, X_test, y_test, X_val, y_val):
    xgb_model = XGBClassifier(
    scale_pos_weight=1.5,
    n_estimators=50,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='binary:logistic',
    random_state=42,
    n_jobs=-1
)

    # Training
    start_time = time.time()
    xgb_model.fit(X_train, y_train)
    training_time = time.time() - start_time
    print(f"Training time (s):{training_time:.2f}")

    # Prediction
    start_time = time.time()
    y_val_pred = xgb_model.predict(X_val)
    y_val_proba = xgb_model.predict_proba(X_val)[:, 1]
    y_test_pred = xgb_model.predict(X_test)
    y_test_proba = xgb_model.predict_proba(X_test)[:, 1]
    prediction_time = time.time() - start_time
    print(f"Prediction time (s):{prediction_time:.2f}")
    return y_test_pred, y_test_proba, y_val_pred, y_val_proba,xgb_model