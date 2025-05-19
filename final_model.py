from preprocessing.preprocessing import CSIC_preprocess, Malicious_phish_preprocess, parsed_request_preprocess
from models.train_model import random_forest_grid_search, random_forest 
from models.evaluate_models import print_metrics, test

from sklearn.model_selection import train_test_split


from config_module.config import CSIC_FILE, PARSE_REQUEST_FILE
from data.raw_data import load


data_raw = load(CSIC_FILE)
data_raw1 = load(PARSE_REQUEST_FILE)
X_resampled, y_resampled = CSIC_preprocess(data_raw)
X_test1, y_test1 = parsed_request_preprocess(data_raw1)


# Train:Val:Test theo tỷ lệ 7:2:1
X_train, X_temp, y_train, y_temp = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.67, random_state=42)

# Train the model
# y_test_pred, y_test_proba, y_val_pred, y_val_proba = random_forest(X_train, y_train, X_test, X_val)
# # Evaluate the model
# print_metrics("Random Forest", y_val, y_val_pred, y_val_proba, "Validation Set")
# print_metrics("Random Forest", y_test, y_test_pred, y_test_proba, "Test Set")   

y_test_pred, y_test_proba, y_val_pred, y_val_proba, best_model = random_forest_grid_search(X_train, y_train, X_test, y_test, X_val, y_val)
print_metrics("Random Forest", y_val, y_val_pred, y_val_proba, "Validation Set")
print_metrics("Random Forest", y_test, y_test_pred, y_test_proba, "Test Set")
print("Best model prediction results:", test(best_model, X_test1, y_test1))

