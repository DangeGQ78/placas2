import cv2
import numpy as np
import pytesseract as pty
from PIL import Image

# Constantes
DIGITOS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890'
UMBRAL_COLOR = 100
UMBRAL_BINARIO = 150
AREA_MIN = 1000
AREA_MAX = 6000
PSM_MODE = 7
PTexto=""
# Opciones de Tesseract
OPTIONS = f"-c tessedit_char_whitelist={DIGITOS} --psm {PSM_MODE}"

# Configuración de Tesseract
pty.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

# Funciones
def extraer_canales_color(region):
    mB = region[:, :, 0]
    mG = region[:, :, 1]
    mR = region[:, :, 2]
    return mB, mG, mR

def procesar_placa(placa):
    alp, anp, _ = placa.shape
    Mva = np.zeros((alp, anp))

    mBp, mGp, mRp = extraer_canales_color(placa)

    for col in range(alp):
        for fil in range(anp):
            Max = max(mRp[col, fil], mGp[col, fil], mBp[col, fil])
            Mva[col, fil] = 255 - Max

    _, bin = cv2.threshold(Mva, UMBRAL_BINARIO, 255, cv2.THRESH_BINARY)
    return bin

# Captura de video
captura = cv2.VideoCapture(1)

while True:
    ret, frame = captura.read()

    if not ret:
        break

    cv2.putText(frame, PTexto[0:7], (420, 430), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 0), 2)

    al, an, c = frame.shape

    x1 = int(an / 3)
    x2 = int(x1 * 2)

    y1 = int(al / 3)
    y2 = int(y1 * 2)

    # cv2.rectangle(frame, (x1 + 160,y1 + 500), (1120, 940),(0,0,0),cv2.FILLED)
    # cv2.putText(frame, 'cargando', (x1 + 100, y1 + 550), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.rectangle(frame, (x1 + 1, y1 + 1), (x2 + 1, y2 + 1), (0, 255, 0), 2)

    recorte = frame[y1:y2, x1:x2]

    mB = np.matrix(recorte[:, :, 0])

    mG = np.matrix(recorte[:, :, 1])

    mR = np.matrix(recorte[:, :, 2])

    color = cv2.absdiff(mG, mB)

    _, umbral = cv2.threshold(color, 40, 255, cv2.THRESH_BINARY)

    contornos, _ = cv2.findContours(umbral, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contornos = sorted(contornos, key=lambda x: cv2.contourArea(x), reverse=True)

    for contorno in contornos:
        area = cv2.contourArea(contorno)
        if area > 1000 and area < 6000:

            x, y, ancho, alto = cv2.boundingRect(contorno)

            xpi = x + x1
            ypi = y + y1

            xpf = x + ancho + x1
            ypf = y + alto + y1

            cv2.rectangle(frame, (xpi, ypi), (xpf, ypf), (255, 255, 0), 2)

            placa = frame[ypi:ypf, xpi:xpf]

            alp, anp, cp = placa.shape

            Mva = np.zeros((alp, anp))

            mBp = np.matrix(placa[:, :, 0])

            mGp = np.matrix(placa[:, :, 1])

            mRp = np.matrix(placa[:, :, 2])

            for col in range(0, alp):
                for fil in range(0, anp):
                    Max = max(mRp[col, fil], mGp[col, fil], mBp[col, fil])
                    Mva[col, fil] = 255 - Max

            _, bin = cv2.threshold(Mva, 150, 255, cv2.THRESH_BINARY)

            bin = bin.reshape(alp, anp)
            bin = Image.fromarray(bin)
            bin = bin.convert("L")
            PTexto = pty.image_to_string(bin,config=OPTIONS)
            


    cv2.imshow('ventana', frame)
    t = cv2.waitKey(1)

    if t == 27:
        break

captura.release()
cv2.destroyAllWindows()