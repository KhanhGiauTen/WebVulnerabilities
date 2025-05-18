
import pandas as pd

def load(link):
    # Load CSIC dataset
    data_csic = pd.read_csv(link, delimiter=',', on_bad_lines='skip')
    data_csic['Accept'] = data_csic['Accept'].fillna(data_csic['Accept'].mode()[0])

    return data_csic