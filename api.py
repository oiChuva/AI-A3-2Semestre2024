import os
import cv2
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from keras.models import load_model  # TensorFlow is required for Keras to work

model = load_model("keras_model.h5", compile=False)
app = FastAPI()

# Função para carregar e processar a IA
def IA(imagem_processada, tipo_p):
    if tipo_p == "pessoa":
        np.set_printoptions(suppress=True)

        with open("labels.txt", "r") as file:
            class_names = file.readlines()

        imagem_processada = cv2.resize(imagem_processada, (224, 224))
        imagem_rgb = cv2.cvtColor(imagem_processada, cv2.COLOR_GRAY2RGB)
        imagem_array = np.asarray(imagem_rgb, dtype=np.float32).reshape(1, 224, 224, 3)
        imagem_array = (imagem_array / 127.5) - 1

        prediction = model.predict(imagem_array)
        index = np.argmax(prediction)
        class_name = class_names[index].strip()
        confidence_score = prediction[0][index]

        if confidence_score * 100 > 65:
            return f"{class_name} Confiança: {confidence_score * 100:.2f}%"
        else:
            print(f"Dados não conclusivos: confiança: {confidence_score * 100:.2f}%")
            return f"Não conseguimos obter conclusões confiáveis sobre a foto, tente novamente com outra foto!"

# Função para tratar a imagem
def image_treat(imagem, filename):
    # Verifica se o diretório de saída existe
    output_dir = './Processed'
    os.makedirs(output_dir, exist_ok=True)

    # Converte para escala de cinza
    imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

    # Suaviza a imagem
    imagem_suavizada = cv2.GaussianBlur(imagem_cinza, (3, 3), 0)

    # Detecta contornos
    contornos_x = cv2.Sobel(imagem_suavizada, cv2.CV_64F, 1, 0)
    contornos_x = np.uint8(np.absolute(contornos_x))
    contornos_y = cv2.Sobel(imagem_suavizada, cv2.CV_64F, 0, 1)
    contornos_y = np.uint8(np.absolute(contornos_y))
    imagem_contornada = cv2.bitwise_or(contornos_x, contornos_y)

    # Salva a imagem processada
    output_path = os.path.join(output_dir, f"Processed_{filename}")
    cv2.imwrite(output_path, imagem_contornada)
    print(f"Imagem {filename} salva em {output_path}")
    return imagem_contornada

# Função para ler e decodificar a imagem enviada
async def read_image(file: UploadFile):
    # Lê os bytes do arquivo
    image_bytes = await file.read()

    # Converte os bytes para uma matriz NumPy
    np_array = np.frombuffer(image_bytes, np.uint8)

    # Decodifica a matriz NumPy em uma imagem
    imagem = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    return imagem

@app.post("/AI-Pensa", response_class=JSONResponse)
async def webhook(
        file: UploadFile = File(...),  # Captura a imagem enviada
        description: str = Form(...),  # Captura a string enviada
):
    """Recebe dados via webhook e processa a inclusão."""

    # Valida se o arquivo enviado é uma imagem
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="O arquivo enviado não é uma imagem.")

    # Lê e processa a imagem
    try:
        imagem = await read_image(file)
        imagem_processada = image_treat(imagem, file.filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao processar a imagem: {e}")

    # Faz a análise com a IA
    try:
        resultado = IA(imagem_processada, description)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao executar a IA: {e}")

    return JSONResponse(content={"message": resultado}, status_code=200)

@app.get("/test-page", response_class=HTMLResponse)
async def test_page():
    html_content = """
        <h1>Server Test Page</h1>
        <p>Server is Running!</p>
        <p>Welcome to the FastAPI Server Test Page</p>
        <p>Status: OK</p>
    """
    return HTMLResponse(content=html_content)
