
import logging
import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

logging.basicConfig(level=logging.DEBUG)

def _modelo_autoencoder(dim: int):
    inp = keras.Input(shape=(dim,))
    x = layers.Dense(dim//2, activation='relu')(inp)
    x = layers.Dense(dim//4, activation='relu')(x)
    x = layers.Dense(dim//2, activation='relu')(x)
    out = layers.Dense(dim, activation='linear')(x)
    model = keras.Model(inp, out)
    model.compile(optimizer='adam', loss='mse')
    return model

def imputar_autoencoder(df: pd.DataFrame, epochs: int = 150) -> pd.DataFrame:
    imp0 = SimpleImputer(strategy='constant', fill_value=0)
    dados = imp0.fit_transform(df)
    scaler = MinMaxScaler()
    dados_norm = scaler.fit_transform(dados)
    model = _modelo_autoencoder(dados_norm.shape[1])
    model.fit(dados_norm, dados_norm, epochs=epochs, batch_size=16, verbose=0)
    recon = model.predict(dados_norm, verbose=0)
    dados_imp = scaler.inverse_transform(recon)
    df_imp = pd.DataFrame(dados_imp, columns=df.columns, index=df.index)
    df_imp[df.notna()] = df[df.notna()]
    logging.debug('Imputação AutoEncoder pronta')
    return df_imp

def imputar_media(df: pd.DataFrame) -> pd.DataFrame:
    imp = SimpleImputer(strategy='mean')
    return pd.DataFrame(imp.fit_transform(df), columns=df.columns, index=df.index)

def imputar_mediana(df: pd.DataFrame) -> pd.DataFrame:
    imp = SimpleImputer(strategy='median')
    return pd.DataFrame(imp.fit_transform(df), columns=df.columns, index=df.index)

def imputar_knn(df: pd.DataFrame, n=5) -> pd.DataFrame:
    imp = KNNImputer(n_neighbors=n)
    return pd.DataFrame(imp.fit_transform(df), columns=df.columns, index=df.index)
