from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

def print_metrics(train_type, y_true, y_pred, y_proba, dataset_name):
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
    
    
def test(model: RandomForestClassifier, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    test_error = 1 - accuracy
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    logloss = log_loss(y_test, y_proba)
   
    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Test Error (1-Accuracy): {test_error:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"Log Loss: {logloss:.4f}")  
    
