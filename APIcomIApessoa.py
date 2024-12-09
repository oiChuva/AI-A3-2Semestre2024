import os.path

from keras.models import load_model  # TensorFlow is required for Keras to work
import cv2
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.responses import JSONResponse

app = FastAPI()

# código da IA vai aqui
def IA(imagem, tipo_p):
  if tipo_p == "pessoa":
    np.set_printoptions(suppress=True)
    # Load the model
    model = load_model("keras_Model.h5", compile=False)
    # Load the labels
    class_names = open("labels.txt", "r").readlines()
    # Make the image a numpy array and reshape it to the models input shape.
    imagem = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

    # Normalize the image array
    imagem = (imagem / 127.5) - 1

    # Predicts the model
    prediction = model.predict(imagem)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Print prediction and confidence score
    print("Class:", class_name[2:], end="")
    print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")
    
    if str((np.round(confidence_score * 100))[:-2], "%") > 65 :
      return "Sim, tem."
    else:
      return "Não tem."

def image_treat(imagem, filename):
    (height, width, depth) = imagem.shape

    # - Escala de CINZA AQUI -------------------------------------------------------------
    imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

    # - Suavização da IMAGEM AQUI
    imagemSuavizada = cv2.GaussianBlur(imagem, (3, 3), 0)

    # - CONTORNOS AQUI
    imagemContornadaX = cv2.Sobel(imagemSuavizada, cv2.CV_64F, 1, 0)
    imagemContornadaX = np.uint8(np.absolute(imagemContornadaX))
    imagemContornadaY = cv2.Sobel(imagemSuavizada, cv2.CV_64F, 0, 1)
    imagemContornadaY = np.uint8(np.absolute(imagemContornadaY))
    imagemContornadaXY = cv2.bitwise_or(imagemContornadaX, imagemContornadaY)

    # Só para ter certeza que esta funcionando - PARA VER SE FUNCIONA, DESCOMENTE A PRÓXIMA LINHA
    # cv2_imshow(imagemContornadaXY)

    output_path = os.path.join('./Processed', f"Processed_{filename}")
    cv2.imwrite(output_path, imagemContornadaXY)
    print(f"Imagem {filename} salva")
    return imagemContornadaXY

async def read_image(file: UploadFile, show=False):
    # Lendo os bytes da imagem enviada
    image_bytes = await file.read()

    # Convertendo os bytes para uma matriz NumPy
    np_array = np.frombuffer(image_bytes, np.uint8)

    # Decodificando a matriz NumPy em uma imagem
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    if show:
        await show_image(image)

    return image


async def show_image(image: np.ndarray):
    # Obtendo as dimensões originais da imagem
    original_height, original_width = image.shape[:2]

    # Definindo um fator de escala (exemplo: 50% do tamanho original)
    scale_percent = 50
    new_width = int(original_width * scale_percent / 100)
    new_height = int(original_height * scale_percent / 100)

    # Alternativamente, defina um tamanho máximo e calcule proporcionalmente
    max_width = 800
    max_height = 600
    if original_width > max_width or original_height > max_height:
        aspect_ratio = original_width / original_height
        if aspect_ratio > 1:  # Largura maior que altura
            new_width = max_width
            new_height = int(max_width / aspect_ratio)
        else:  # Altura maior ou igual à largura
            new_height = max_height
            new_width = int(max_height * aspect_ratio)

    # Redimensionando a imagem mantendo a proporção
    resized_image = cv2.resize(image, (new_width, new_height))

    # Mostrando a imagem redimensionada
    cv2.imshow("Imagem", resized_image)
    cv2.waitKey(0)
    return


@app.post("/AI-Pensa", response_class=JSONResponse)
async def webhook(
        file: UploadFile = File(),  # Captura a imagem enviada
        description: str = Form()  # Captura a string enviada
):
    """Recebe dados via webhook e processa a inclusão."""

    # Validando parâmetros
    if file is None:
        raise HTTPException(status_code=400, detail="Foto não fornecida.")

    tipo_p = description
    if tipo_p is None:
        raise HTTPException(status_code=400, detail="Tipo não fornecido.")

    # Validando se é uma imagem
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="O arquivo enviado não é uma imagem.")

    # Lendo file como imagem
    imagem = await read_image(file, False)

    # Processando imagem
    try:
        imagem_processada = image_treat(imagem, file.filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao processar imagem: {e}")

    # dá a imagem processada para a IA resolver se tem o item ou não
    try:
        resultado = IA(imagem_processada, tipo_p)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ia com erro ao receber e processar imagem: {e}")

    return JSONResponse(content={"message": resultado}, status_code=200)
    # tipo é o que a IA vai analisar da foto, pessoa, faca, cadeira, lixeira, por exemplo. _p de pergunta


@app.get("/test-page", response_class=HTMLResponse)
async def test_page():
    html_content = """
        Server Test Page </br>
         </br>
        Server is Running! </br>
        Welcome to the FastAPI Server Test Page </br>
        Status: OK </br>
    """
    return HTMLResponse(content=html_content)
