
import logging
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif

logging.basicConfig(level=logging.DEBUG)

def selecionar_k_melhores(X: pd.DataFrame, y: pd.Series, k: int = 10) -> pd.DataFrame:
    sel = SelectKBest(score_func=f_classif, k=k)
    X_new = sel.fit_transform(X, y)
    cols = X.columns[sel.get_support()]
    logging.debug(f'Features escolhidas: {cols.tolist()}')
    return pd.DataFrame(X_new, columns=cols, index=X.index)
