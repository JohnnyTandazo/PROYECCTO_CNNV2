import streamlit as st
import tensorflow as tf
import json
import numpy as np
from streamlit_webrtc import webrtc_streamer
import av
import cv2
from PIL import Image

# Cargar diccionario de clases
with open('class_indices.json', 'r') as f:
    class_indices = json.load(f)
indices_to_class = {v: k for k, v in class_indices.items()}

# Cargar modelo SavedModel
model = tf.saved_model.load("modelov2_prendas_savedmodel")
infer = model.signatures["serving_default"]
output_tensor_name = 'output_0'  # Cambia si tu output tiene otro nombre

def predict_image(img_array):
    input_tensor = tf.constant(np.expand_dims(img_array, axis=0), dtype=tf.float32)
    preds = infer(input_tensor)[output_tensor_name].numpy()
    pred_index = np.argmax(preds)
    pred_class = indices_to_class.get(pred_index, "Unknown")
    confidence = preds[0][pred_index]
    return pred_class, confidence

def preprocess_image(image):
    img = image.resize((224, 224))
    img_array = np.array(img).astype(np.float32) / 255.0
    if img_array.shape[-1] == 4:  # Si PNG tiene canal alpha, eliminarlo
        img_array = img_array[..., :3]
    return img_array

def predict_frame(frame):
    img = frame.to_ndarray(format="rgb24")
    img_resized = cv2.resize(img, (224, 224))
    img_norm = img_resized.astype(np.float32) / 255.0

    pred_class, confidence = predict_image(img_norm)

    cv2.putText(img_resized, f"{pred_class}: {confidence:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return av.VideoFrame.from_ndarray(img_resized, format="rgb24")

st.title("Clasificador de prendas")

# Selector para elegir método de entrada
option = st.radio("Seleccione método de entrada:", ("Cámara en tiempo real", "Subir imagen"))

if option == "Cámara en tiempo real":
    webrtc_streamer(
        key="example",
        video_frame_callback=predict_frame,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
else:
    uploaded_file = st.file_uploader("Sube una imagen para clasificar", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image_pil = Image.open(uploaded_file)
        st.image(image_pil, caption="Imagen subida", use_container_width=True)

        img_array = preprocess_image(image_pil)
        pred_class, confidence = predict_image(img_array)

        st.write(f"**Predicción:** {pred_class}")
        st.write(f"**Confianza:** {confidence:.2f}")
