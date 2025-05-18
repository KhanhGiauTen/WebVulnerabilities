# File paths

# Vectorizer
MAX_FEATURE = 1000

# Training params
TEST_SIZE = 0.2
RANDOM_STATE = 42
CLASS_WEIGHT = 'balanced'
N_ESTIMATORS = 100
MAX_ITER = 10000



# Preprocessing
PCA_COMPONENT = 2

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