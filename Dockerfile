# Utiliza uma imagem base leve do Python
FROM python:3.11-slim

# Define o diretório de trabalho dentro do container
WORKDIR /app

# Copia o arquivo de dependências e instala as bibliotecas necessárias
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copia todo o código da aplicação (incluindo a pasta app e Model)
COPY . .

# Expõe a porta 80 para acesso à API
EXPOSE 80

# Inicia o servidor Uvicorn, apontando para a aplicação FastAPI contida em app/main.py
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
