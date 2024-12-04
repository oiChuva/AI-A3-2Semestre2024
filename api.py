import os.path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.responses import JSONResponse

app = FastAPI()


# if not path.exists('./NoProcessing'):
#     os.mkdir('./NoProcessing')
#
# if not path.exists('/Processed'):
#     os.mkdir('./Processed')


def image_treat(imagem, filename):
    if imagem is not None:
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


# código da IA vai aqui
def IA(imagem, tipo_p):
    return


async def read_image(file: UploadFile, show=False):
    # Lê os bytes da imagem enviada
    image_bytes = await file.read()

    # Converte os bytes para uma matriz NumPy
    np_array = np.frombuffer(image_bytes, np.uint8)

    # Decodifica a matriz NumPy em uma imagem
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    if show:
        await show_image(image)

    return image


async def show_image(image: np.ndarray):
    # Obtem as dimensões originais da imagem
    original_height, original_width = image.shape[:2]

    # Define um fator de escala (exemplo: 50% do tamanho original)
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

    # Redimensiona a imagem mantendo a proporção
    resized_image = cv2.resize(image, (new_width, new_height))

    # Mostra a imagem redimensionada
    cv2.imshow("Imagem", resized_image)
    cv2.waitKey(0)
    return


@app.post("/AI-Pensa")
async def webhook(
        file: UploadFile = File(),  # Captura a imagem enviada
        description: str = Form()  # Captura a string enviada
):
    """Recebe dados via webhook e processa a inclusão."""
    imagem = await read_image(file, False)

    try:
        # data = await request.json()
        print(f"Received data: {description}")
        if imagem is None:
            raise HTTPException(status_code=400, detail="Foto não fornecida.")

        tipo_p = description
        if tipo_p is None:
            raise HTTPException(status_code=400, detail="Tipo não fornecido.")

        # processa imagem
        imagem_processada = image_treat(imagem, file.filename)
        if "error" in imagem_processada:
            raise HTTPException(status_code=400, detail=imagem_processada["error"])

        # dá a imagem processada para a IA resolver se tem o item ou não
        resultado = IA(imagem_processada, tipo_p)
        if "error" in resultado:
            raise HTTPException(status_code=400, detail=resultado["error"])

        return JSONResponse(content={resultado})
        # tipo é o que a IA vai analisar da foto, pessoa, faca, cadeira, lixeira, por exemplo. _p de pergunta
        # foto é o que vem da requisição, imagem é o que vai ser tratado e a IA vai analisar.

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
