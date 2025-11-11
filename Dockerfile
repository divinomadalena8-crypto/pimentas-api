# Use uma imagem base do Python 3.10
FROM python:3.10-slim

# Defina a pasta de trabalho
WORKDIR /code

# Copie o arquivo de requisitos primeiro
COPY requirements.txt .

# Instale os requisitos (vai funcionar aqui)
RUN pip install --no-cache-dir -r requirements.txt

# Copie todo o resto do seu projeto
COPY . .

# Diga ao Hugging Face que seu app vai rodar na porta 7860
EXPOSE 7860

# O comando para ligar o seu main.py
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
