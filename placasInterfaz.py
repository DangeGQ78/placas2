import cv2
import numpy as np
import pytesseract as pty
from PIL import Image, ImageTk
import skimage
import tkinter as tk
from tkinter import filedialog

def procesar_imagen(ruta_imagen):
    global PTexto
    digitos = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890'
    option = "-c tessedit_char_whitelist={}".format(digitos)
    option += " --psm {}".format(7)
    pty.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

    frame = cv2.imread(ruta_imagen)
    
    # Resto del código...
     

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
    print("asad")
    for contorno in contornos:
        area = cv2.contourArea(contorno)
        if area > 2000 and area < 14000:
            
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

            
            _, bin = cv2.threshold(Mva, 160, 255, cv2.THRESH_BINARY)
            
            bin = bin.reshape(alp, anp)
           
            bin = skimage.segmentation.clear_border(bin)
           
            bin = Image.fromarray(bin)
           
            bin = bin.convert("L")
            
            PTexto = pty.image_to_string(bin,config=option)
    # Tu código para procesar la imagen...

    # Ejemplo de cómo actualizar la interfaz con el resultado
    resultado_label.config(text=PTexto)

    # Mostrar la imagen procesada en un widget Label
    imagen_procesada = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    imagen_procesada = imagen_procesada.resize((400, 400), Image.ADAPTIVE)
    imagen_procesada = ImageTk.PhotoImage(imagen_procesada)
    imagen_procesada_label.config(image=imagen_procesada)
    imagen_procesada_label.image = imagen_procesada

# Función para abrir un cuadro de diálogo y seleccionar una imagen
def abrir_imagen():
    ruta_imagen = filedialog.askopenfilename(filetypes=[("Archivos de imagen", "*.png;*.jpg;*.jpeg")])
    if ruta_imagen:
        procesar_imagen(ruta_imagen)

# Crear la interfaz de usuario
app = tk.Tk()
app.title("Procesamiento de Imágenes")

# Botón para abrir una imagen
boton_abrir = tk.Button(app, text="Abrir Imagen", command=abrir_imagen)
boton_abrir.pack(pady=10)

# Widget Label para mostrar la imagen procesada
imagen_procesada_label = tk.Label(app)
imagen_procesada_label.pack(pady=10)

# Widget Label para mostrar el resultado del OCR
resultado_label = tk.Label(app, text="")
resultado_label.pack(pady=10)

# Ejecutar la aplicación
app.mainloop()