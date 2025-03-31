# API de Previsão de Intensidade de Carbono

Este projeto implementa uma API RESTful utilizando FastAPI para realizar previsões da intensidade de carbono. A API consome dados em tempo real da Electricity Maps API, normaliza esses dados com um scaler treinado (MinMaxScaler) e realiza a previsão utilizando um modelo LSTM previamente treinado.

## Funcionalidades

- **Obtenção de Dados:** Consome os últimos 24 registros de intensidade de carbono via API externa.
- **Normalização:** Utiliza um scaler treinado para normalizar os dados recebidos.
- **Previsão:** Realiza a previsão das próximas 24 horas com um modelo LSTM.
- **API REST:** Endpoints para checagem de status (`/`) e para previsão (`/predict`).
- **Docker:** Projeto preparado para ser containerizado e facilmente implantado em ambientes como AWS.

## Estrutura do Projeto

```plaintext
carbon_intensity_prediction_project/
├── Dockerfile
├── README.md
├── requirements.txt
├── app/
│   └── main.py
└── Model/
    ├── LSTM_LCA_Model.keras
    └── minmax_scaler_CI.pkl
