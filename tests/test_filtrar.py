
from src.data_loading import filtrar_pseudometabolico
import pandas as pd

def test_filtrar():
    df = pd.DataFrame({
        'Diagnostico_Real': ['DISTROFIA', 'DISTROFIA COM MANIFESTAÇÃO PSEUDOMETABÓLICA'],
        'Diagnostico_Tipo': [1, None]
    })
    res = filtrar_pseudometabolico(df)
    assert res.shape[0] == 1
