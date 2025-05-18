
import logging, os
import pandas as pd
from sklearn.model_selection import train_test_split
from src.data_loading import carregar_dados, filtrar_pseudometabolico
from src.preprocessing import limpar_dados
from src.imputation import imputar_autoencoder, imputar_media, imputar_mediana, imputar_knn
from src.feature_selection import selecionar_k_melhores
from src.sampling import aplicar_smote
from src.augmentation import ruido_gaussiano
from src.models import lista_modelos
from src.evaluation import avaliar

logging.basicConfig(level=logging.DEBUG)

IMPUTACOES = {
    'AutoEncoder': imputar_autoencoder,
    'Media': imputar_media,
    'Mediana': imputar_mediana,
    'KNN': imputar_knn
}

def pipeline(caminho_csv: str, pasta_out: str):
    df = carregar_dados(caminho_csv)
    df = filtrar_pseudometabolico(df)
    df = limpar_dados(df)
    y = df['Diagnostico_Tipo'].astype(int)
    X = df.drop(columns=['Diagnostico_Tipo'])
    for nome, func in IMPUTACOES.items():
        X_imp = func(X)
        X_sel = selecionar_k_melhores(X_imp, y)
        X_bal, y_bal = aplicar_smote(X_sel, y)
        X_train, X_test, y_train, y_test = train_test_split(
            X_bal, y_bal, test_size=0.1, stratify=y_bal, random_state=42
        )
        alvo = y_bal.value_counts().max()
        X_test_aug, y_test_aug = ruido_gaussiano(X_test, y_test, alvo)
        modelos = lista_modelos()
        resultados = avaliar(modelos, X_train, y_train, X_test_aug, y_test_aug)
        os.makedirs(pasta_out, exist_ok=True)
        resultados.to_csv(os.path.join(pasta_out, f'resultados_{nome}.csv'), index=False)
        logging.info(f'Resultados salvos: {nome}')

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--csv', required=True)
    p.add_argument('--out', default='resultados')
    args = p.parse_args()
    pipeline(args.csv, args.out)
