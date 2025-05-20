from preprocessing.preprocessing import CSIC_preprocess, parsed_request_preprocess
from models.train_model import random_forest_grid_search, linear_svc, xgboost, naive_bayes_opt_gs, decision_tree, knn
from models.evaluate_models import print_metrics, test

from sklearn.model_selection import train_test_split


from config_module.config import CSIC_FILE, PARSE_REQUEST_FILE
from data.raw_data import load

from config_module.config import RANDOM_STATE, TEST_SIZE_1, TEST_SIZE_2


data_raw = load(CSIC_FILE)
data_raw1 = load(PARSE_REQUEST_FILE)
X_resampled, y_resampled = CSIC_preprocess(data_raw)
X_test1, y_test1 = parsed_request_preprocess(data_raw1)


# Train:Val:Test theo tỷ lệ 7:2:1
X_train, X_temp, y_train, y_temp = train_test_split(X_resampled, y_resampled, test_size=TEST_SIZE_1, random_state=RANDOM_STATE)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=TEST_SIZE_2, random_state=RANDOM_STATE)

# Train the model
# y_test_pred, y_test_proba, y_val_pred, y_val_proba = random_forest(X_train, y_train, X_test, X_val)
# # Evaluate the model
# print_metrics("Random Forest", y_val, y_val_pred, y_val_proba, "Validation Set")
# print_metrics("Random Forest", y_test, y_test_pred, y_test_proba, "Test Set")   
def train_test_rf():
    y_test_pred, y_test_proba, y_val_pred, y_val_proba, best_model = random_forest_grid_search(X_train, y_train, X_test, X_val)
    print_metrics("Random Forest", y_val, y_val_pred, y_val_proba, "Validation Set")
    print_metrics("Random Forest", y_test, y_test_pred, y_test_proba, "Test Set")
    test(best_model, X_test1, y_test1)
    
def train_test_linear():
    y_test_pred, y_test_proba, y_val_pred, y_val_proba, best_model = linear_svc(X_train, y_train, X_test, X_val)
    print_metrics("LinearSVC", y_val, y_val_pred, y_val_proba, "Validation Set")
    print_metrics("LinearSVC", y_test, y_test_pred, y_test_proba, "Test Set")
    test(best_model, X_test1, y_test1)

def train_test_xgboost():
    y_test_pred, y_test_proba, y_val_pred, y_val_proba, best_model = xgboost(X_train, y_train, X_test, X_val)
    print_metrics("XGboot", y_val, y_val_pred, y_val_proba, "Validation Set")
    print_metrics("XGboot", y_test, y_test_pred, y_test_proba, "Test Set")
    test(best_model, X_test1, y_test1)

def train_test_naive_bayes():
    y_test_pred, y_test_proba, y_val_pred, y_val_proba, best_model = naive_bayes_opt_gs(X_train, y_train, X_test, X_val)
    print_metrics("Naive Bayes", y_val, y_val_pred, y_val_proba, "Validation Set")
    print_metrics("Naive Bayes", y_test, y_test_pred, y_test_proba, "Test Set")
    test(best_model, X_test1, y_test1)

def train_test_decision_tree():
    y_test_pred, y_test_proba, y_val_pred, y_val_proba, best_model = decision_tree(X_train, y_train, X_test, X_val)
    print_metrics("Decision Tree", y_val, y_val_pred, y_val_proba, "Validation Set")
    print_metrics("Decision Tree", y_test, y_test_pred, y_test_proba, "Test Set")
    test(best_model, X_test1, y_test1)

def train_test_knn():
    y_test_pred, y_test_proba, y_val_pred, y_val_proba, best_model = knn(X_train, y_train, X_test, X_val)
    print_metrics("KNN", y_val, y_val_pred, y_val_proba, "Validation Set")
    print_metrics("KNN", y_test, y_test_pred, y_test_proba, "Test Set")
    test(best_model, X_test1, y_test1)

if __name__ == "__main__":
    train_test_rf()
    train_test_linear()
    train_test_xgboost()
    train_test_naive_bayes()
    train_test_decision_tree()
    train_test_knn()


