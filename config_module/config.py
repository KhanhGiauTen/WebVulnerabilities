# File paths

# Vectorizer
MAX_FEATURE = 1000


# Training params
TEST_SIZE_1 = 0.3
TEST_SIZE_2 = 0.67
RANDOM_STATE = 42

MAX_ITER = 10000
VERBOSE = 1
N_JOBS = 2

#RANDOM FOREST
N_ESTIMATORS = 100


#RANDOM FOREST GRID SEARCH
GRID_SEARCH_N_ESTIMATORS = [50, 100]  # Số lượng cây trong rừng
MAX_DEPTH = [10, None]  # Độ sâu tối đa của cây
MIN_SAMPLES_SPLIT = [2, 5]  # Số mẫu tối thiểu để chia một node
CV = 3  # Số lần gập trong cross-validation




#KNN
KNN_N_NEIGHBORS = 5
KNN_METRIC = 'euclidean'


#XGBoost
XGBOOST_SCALE_POSITIVE_WEIGHT = 1.5
XGBOOST_N_ESTIMATORS = 50
XGBOOST_MAX_DEPTH = 6
XGBOOST_LEARNING_RATE = 0.1
XGBOOST_SUBSAMPLE = 0.8
XGBOOST_COLSAMPLE_BYTREE = 0.8
OBJECTIVE = 'binary:logistic'


# LINEAR SVC
LINEAR_SVC_C = 1.0
LINEAR_SVC_TOL = 0.01
LINEAR_SVC_MAX_ITER = 1000
LINEAR_SVC_CV = 5



# Preprocessing
PCA_COMPONENT = 300

# Data paths
import os

import os

# Lấy thư mục config (chứa config.py)
CONFIG_DIR = os.path.dirname(__file__)

# Lấy thư mục cha của config (thư mục gốc dự án)
BASE_DIR = os.path.dirname(CONFIG_DIR)

# Đường dẫn tới thư mục data ở cùng cấp với config
DATA_DIR = os.path.join(BASE_DIR, "data")

CSIC_FILE = os.path.join(DATA_DIR, "csic_database.csv")

PARSE_REQUEST_TEST = os.path.join(DATA_DIR, "parsed_requests_test.csv")

PARSE_REQUEST_TRAIN = os.path.join(DATA_DIR, "parsed_requests_train.csv")