from fastapi import FastAPI, HTTPException
import requests
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import os
from fastapi.responses import JSONResponse
import logging

# Configuração do logging para auxiliar na depuração
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="API de Previsão de Intensidade de Carbono")

# Configurações da API do Electricity Maps
API_KEY = 'czG7nq1wv9OHi1phrXUn'  # Chave da API fornecida
REGION = 'PT'  # Código da região (Portugal)
URL = f'https://api.electricitymap.org/v3/carbon-intensity/history?zone={REGION}'
HEADERS = {'auth-token': API_KEY}

# Constante para o nome da coluna utilizada no treinamento do scaler
COLUMN_TRAIN = "Carbon Intensity gCO₂eq/kWh (LCA)"

@app.on_event("startup")
def load_resources():
    """
    Carrega os recursos necessários (scaler e modelo) no início da execução da API.
    Os recursos são armazenados em app.state para facilitar o acesso nos endpoints.
    """
    try:
        scaler_path = os.path.join(os.getcwd(), 'Model', 'minmax_scaler_CI.pkl')
        app.state.scaler = joblib.load(scaler_path)
        logger.info("Scaler carregado com sucesso.")
    except Exception as e:
        logger.error("Erro ao carregar o scaler: %s", e)
        raise e

    try:
        model_path = os.path.join(os.getcwd(), 'Model', 'LSTM_LCA_Model.keras')
        app.state.model = tf.keras.models.load_model(model_path)
        logger.info("Modelo carregado com sucesso.")
    except Exception as e:
        logger.error("Erro ao carregar o modelo: %s", e)
        raise e

def obter_dados_api():
    """
    Obtém os últimos 24 registros de dados da API externa e retorna um DataFrame.
    """
    response = requests.get(URL, headers=HEADERS)
    if response.status_code == 200:
        data = response.json()['history']
        df = pd.DataFrame(data)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values(by='datetime', ascending=True)
        df = df[['datetime', 'carbonIntensity']]
        df.rename(columns={'carbonIntensity': 'LCA'}, inplace=True)
        return df.tail(24)
    else:
        raise HTTPException(status_code=response.status_code, detail="Erro ao acessar API externa")

def normalizar_dados(df):
    """
    Normaliza os dados utilizando o scaler previamente treinado.
    Realiza o mapeamento da coluna 'LCA' para o nome utilizado no treinamento e retorna
    o DataFrame com a coluna renomeada de volta para 'LCA'.
    """
    df = df.rename(columns={'LCA': COLUMN_TRAIN})
    df[COLUMN_TRAIN] = app.state.scaler.transform(df[[COLUMN_TRAIN]])
    df = df.rename(columns={COLUMN_TRAIN: 'LCA'})
    return df

def fazer_previsao(dados):
    """
    Formata os dados para a LSTM, realiza a previsão e retorna a classe com maior probabilidade.
    """
    dados_formatados = dados['LCA'].values.reshape(1, 24, 1)
    previsao = app.state.model.predict(dados_formatados)
    previsao_classe = int(np.argmax(previsao, axis=1)[0])
    return previsao_classe

@app.get("/")
def root():
    """
    Endpoint raiz que confirma que a API está em execução.
    """
    return {"message": "API de Previsão de Intensidade de Carbono em Execução"}

@app.get("/predict")
def predict():
    try:
        # Obtém dados ao vivo
        df_dados = obter_dados_api()
        # Se não tiver sido feito na função obter_dados_api, converta a coluna datetime para string:
        df_dados['datetime'] = df_dados['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
        # Normaliza os dados
        df_normalizado = normalizar_dados(df_dados)
        # Realiza a previsão
        previsao = fazer_previsao(df_normalizado)
        result = {
            "prediction": previsao,
            "data": df_dados.to_dict(orient="records")
        }
        return JSONResponse(content=result)
    except Exception as e:
        logger.error("Erro na previsão: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

