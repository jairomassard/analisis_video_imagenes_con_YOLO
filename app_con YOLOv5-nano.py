import streamlit as st
import cv2
import torch
from PIL import Image
import numpy as np
import tempfile
import os

# Cargar el modelo YOLOv5 desde torch hub
model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)

st.title("Detección de objetos con Modelo YOLOv5")

# Elegir el tipo de detección
detection_type = st.selectbox(
    "Selecciona el tipo de detección",
    ("Cargar Imagen", "Cámara Web", "Cargar Video")
)

# Detección en tiempo real con la cámara web
if detection_type == "Cámara Web":
    st.write("Usando la cámara web para detección en tiempo real")

    # Capturar la cámara web
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("No se pudo acceder a la cámara web.")
    else:
        stframe = st.empty()
        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("No se pudo leer el flujo de la cámara.")
                break

            # Realizar la detección con YOLOv5
            results = model(frame)
            frame = np.squeeze(results.render())  # Renderizar las detecciones en la imagen

            # Mostrar la imagen con las detecciones
            stframe.image(frame, channels="BGR", use_column_width=True)

        cap.release()

# Detección en una imagen cargada
elif detection_type == "Cargar Imagen":
    uploaded_image = st.file_uploader("Cargar imagen", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Imagen cargada", use_column_width=True)

        # Convertir imagen a formato numpy
        img_np = np.array(image)

        # Realizar la detección
        results = model(img_np)
        results_image = np.squeeze(results.render())

        # Mostrar la imagen con las detecciones
        st.image(results_image, caption="Imagen con detecciones", use_column_width=True)

# Detección en un video cargado
elif detection_type == "Cargar Video":
    uploaded_video = st.file_uploader("Cargar video", type=["mp4", "mov", "avi", "mkv"])
    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)  # Crear archivo temporal
        tfile.write(uploaded_video.read())
        tfile.flush()

        # Abrir el video
        cap = cv2.VideoCapture(tfile.name)
        
        # Preparar el escritor de video para guardar el archivo procesado
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        output_path = os.path.join(tempfile.gettempdir(), 'output_video.mp4')

        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.warning("El video ha terminado.")
                break

            # Realizar la detección en cada cuadro del video
            results = model(frame)
            frame = np.squeeze(results.render())  # Renderizar las detecciones en la imagen

            # Guardar el cuadro procesado en el archivo de salida
            out.write(frame)

            # Mostrar el cuadro con las detecciones
            stframe.image(frame, channels="BGR", use_column_width=True)

        cap.release()
        out.release()

        # Mostrar botón para descargar el video procesado
        with open(output_path, 'rb') as f:
            st.download_button('Descargar video procesado', f, file_name='output_video.mp4', mime='video/mp4')


# la aplicacion se ejecuta poniendo esto en la terminal en el ambiente virtual activado:
# streamlit run app.py
