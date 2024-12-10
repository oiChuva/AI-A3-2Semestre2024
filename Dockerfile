# Use uma imagem base do Python com suporte para pip
FROM python:3.10.12

# Define o diretório de trabalho dentro do container
WORKDIR /app

# Criando diretórios
RUN mkdir NoProcessing
RUN mkdir Processed

# Instala as dependências do sistema necessárias para o OpenCV
RUN apt-get update && apt-get install -y \
    libglib2.0-0 libsm6 libxrender1 libxext6 ffmpeg libgl1-mesa-glx && \
    rm -rf /var/lib/apt/lists/*

# Instala as dependências do Python listadas no arquivo requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir tensorflow==2.10.0 keras fastapi uvicorn opencv-python-headless numpy==1.26.4 python-multipart

# Copia os arquivos necessários para o container
COPY api.py /app
COPY api.py /app
COPY keras_model.h5 /app
COPY labels.txt /app

# Expõe a porta onde o servidor irá rodar
EXPOSE 5000

# Comando para rodar o servidor FastAPI com o Uvicorn
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "5000"]