import streamlit as st
import tensorflow as tf
import json
import numpy as np
from streamlit_webrtc import webrtc_streamer
import av

# 1. Cargar diccionario de clases
with open('class_indices.json', 'r') as f:
    class_indices = json.load(f)
indices_to_class = {v: k for k, v in class_indices.items()}

# 2. Cargar modelo SavedModel
model = tf.saved_model.load("modelov2_prendas_savedmodel")
infer = model.signatures["serving_default"]
output_tensor_name = 'output_0'  # Cambia si tu output tiene otro nombre

# 3. Función para predecir a partir de frame de video (imagen)
def predict_frame(frame):
    # Convertir frame (PyAV VideoFrame) a numpy array
    img = frame.to_ndarray(format="rgb24")
    # Redimensionar a (224,224)
    import cv2
    img_resized = cv2.resize(img, (224, 224))
    img_norm = img_resized.astype(np.float32) / 255.0
    input_tensor = tf.constant(np.expand_dims(img_norm, axis=0))
    
    preds = infer(input_tensor)[output_tensor_name].numpy()
    pred_index = np.argmax(preds)
    pred_class = indices_to_class.get(pred_index, "Unknown")
    confidence = preds[0][pred_index]

    # Puedes mostrar texto sobre el frame con OpenCV si quieres (opcional)
    cv2.putText(img_resized, f"{pred_class}: {confidence:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return av.VideoFrame.from_ndarray(img_resized, format="rgb24")

# 4. Streamlit webRTC streamer
def video_frame_callback(frame):
    return predict_frame(frame)

st.title("Clasificador de prendas en tiempo real")

webrtc_streamer(
    key="example",
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# 5. Además puedes añadir uploader para imagenes estáticas, pero eso sería aparte
