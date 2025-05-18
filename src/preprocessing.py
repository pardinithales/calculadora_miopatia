
import logging
import pandas as pd

logging.basicConfig(level=logging.DEBUG)

DESCARTAR_FIXAS = ['PacienteID', 'Diagnostico_Real']

def remover_colunas_na(df: pd.DataFrame, limite: float = 0.3) -> pd.DataFrame:
    proporcao = df.isna().mean()
    remover = proporcao[proporcao > limite].index.tolist()
    logging.debug(f'Colunas descartadas por NA: {remover}')
    return df.drop(columns=remover)

def limpar_dados(df: pd.DataFrame) -> pd.DataFrame:
    df = remover_colunas_na(df)
    df = df.drop(columns=DESCARTAR_FIXAS, errors='ignore')
    logging.debug(f'Colunas restantes: {df.columns.tolist()}')
    return df
