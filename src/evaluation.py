
import logging
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize

logging.basicConfig(level=logging.DEBUG)

def _auc(y_true, y_prob, classes):
    y_bin = label_binarize(y_true, classes=classes)
    return roc_auc_score(y_bin, y_prob, average='macro')

def avaliar(modelos: dict, X_train, y_train, X_test, y_test) -> pd.DataFrame:
    resultados = []
    classes = sorted(np.unique(y_train))
    for nome, modelo in modelos.items():
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)
        if hasattr(modelo, 'predict_proba'):
            y_prob = modelo.predict_proba(X_test)
            auc = _auc(y_test, y_prob, classes)
        else:
            auc = np.nan
        resultados.append({
            'Modelo': nome,
            'Acuracia': accuracy_score(y_test, y_pred),
            'Precisao': precision_score(y_test, y_pred, average='macro', zero_division=0),
            'Sensibilidade': recall_score(y_test, y_pred, average='macro', zero_division=0),
            'F1': f1_score(y_test, y_pred, average='macro', zero_division=0),
            'AUC': auc
        })
    return pd.DataFrame(resultados)
