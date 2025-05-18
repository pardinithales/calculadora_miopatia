from fastapi import FastAPI
from pydantic import BaseModel
from src.models import lista_modelos
import numpy as np

app = FastAPI()

class PredictRequest(BaseModel):
    modelo: str
    features: list

@app.post("/predict")
def predict(req: PredictRequest):
    modelos = lista_modelos()
    if req.modelo not in modelos:
        return {"erro": f"Modelo '{req.modelo}' não encontrado."}
    modelo = modelos[req.modelo]
    # Exemplo: modelo precisa ser treinado antes de prever. Aqui, apenas para teste, simulamos um fit.
    # Em produção, carregue um modelo já treinado.
    X_dummy = np.array([req.features, req.features])
    y_dummy = [0, 1]
    modelo.fit(X_dummy, y_dummy)
    pred = modelo.predict([req.features])
    return {"predicao": int(pred[0])} 