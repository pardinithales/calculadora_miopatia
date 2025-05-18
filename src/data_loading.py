
import logging
import pandas as pd

logging.basicConfig(level=logging.DEBUG)

def carregar_dados(caminho_csv: str) -> pd.DataFrame:
    """Carrega o arquivo CSV em um DataFrame."""
    logging.debug(f'Carregando dados de {caminho_csv}')
    return pd.read_csv(caminho_csv)

def filtrar_pseudometabolico(df: pd.DataFrame) -> pd.DataFrame:
    """Remove pacientes pseudometabólicos e registros sem classe."""
    filtro = df['Diagnostico_Real'].str.contains('PSEUDOMETAB', na=False)
    df = df.loc[~filtro].copy()
    df = df.dropna(subset=['Diagnostico_Tipo'])
    logging.debug(f'Tamanho após filtro: {df.shape}')
    return df
