
import logging
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.DEBUG)

def ruido_gaussiano(X: pd.DataFrame, y: pd.Series, alvo: int):
    X_out = X.copy()
    y_out = y.copy()
    for classe, freq in y.value_counts().items():
        falta = alvo - freq
        if falta <= 0:
            continue
        amostras = X[y == classe].sample(n=falta, replace=True, random_state=42)
        ruido = np.random.normal(0, 0.01, size=amostras.shape)
        novos = amostras.values + ruido
        X_out = pd.concat([X_out, pd.DataFrame(novos, columns=X.columns)])
        y_out = pd.concat([y_out, pd.Series([classe]*falta)])
    return X_out, y_out
