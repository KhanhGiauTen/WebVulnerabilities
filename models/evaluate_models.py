from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score, log_loss
import matplotlib.pyplot as plt

# import train_model
# from train_model import rf_model, svm_model, X_train_scaled, y_train, X_test_scaled, y_test

# def evaluate_model_rf(model, X_test, y_test):
#     y_pred = model.predict(X_test)
#     y_probs = model.predict_proba(X_test)[:, 1] #lấy xác suất các mẫu có lỗ hổng
#     fpr, tpr, thresholds = roc_curve(y_test, y_probs)
#     print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
#     print(f"Precision: {precision_score(y_test, y_pred, average='weighted'):.4f}")
#     print(f"Recall: {recall_score(y_test, y_pred, average='weighted'):.4f}")
#     print(f"F1-score: {f1_score(y_test, y_pred, average='weighted'):.4f}")
#     print(f"roc_auc: {roc_auc_score(y_test, y_probs)}")
#     # Vẽ ROC Curve
#     plt.figure()
#     plt.plot(fpr, tpr, color='b', lw=2, label=f'ROC curve (AUC = {roc_auc_score(y_test, y_probs):.2f})')
#     plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Đường phân loại ngẫu nhiên
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('False Positive Rate (FPR)')
#     plt.ylabel('True Positive Rate (TPR)')
#     plt.title('ROC Curve')
#     plt.legend(loc='lower right')
#     plt.show()
    
# print("Random Forest Evaluation:")
# evaluate_model_rf(rf_model, X_test, y_test)


# def evaluate_model_svm(model, X_test, y_test):
#     y_pred = model.predict(X_test)
#     print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
#     print(f"Precision: {precision_score(y_test, y_pred, average='weighted'):.4f}")
#     print(f"Recall: {recall_score(y_test, y_pred, average='weighted'):.4f}")
#     print(f"F1-score: {f1_score(y_test, y_pred, average='weighted'):.4f}")
    
#     # Vẽ đồ thị
#     plt.figure(figsize=(8, 6))
#     plt.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=y_train, cmap='coolwarm', edgecolors='k')

#     # Tính đường quyết định (decision boundary)
#     w = svm_model.coef_[0]
#     b = svm_model.intercept_[0]
#     x_vals = np.linspace(X_train_scaled[:, 0].min(), X_train_scaled[:, 0].max(), 100)
#     y_vals = -(w[0] * x_vals + b) / w[1]

#     plt.plot(x_vals, y_vals, 'k--', label='Decision Boundary')
#     plt.xlabel("Feature 1")
#     plt.ylabel("Feature 2")
#     plt.title("LinearSVC Decision Boundary (2D)")
#     plt.legend()
#     plt.show()
    
# print("\nSVM Evaluation:")
# evaluate_model_svm(svm_model, X_test_scaled, y_test)

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
