
import logging
import pandas as pd
from imblearn.over_sampling import SMOTE

logging.basicConfig(level=logging.DEBUG)

def aplicar_smote(X: pd.DataFrame, y: pd.Series):
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)
    logging.debug(f'SMOTE produziu {X_res.shape}')
    return X_res, y_res
