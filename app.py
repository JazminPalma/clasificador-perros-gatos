import json
from pathlib import Path
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

st.set_page_config(page_title="Perros vs Gatos IA", layout="centered")
st.title("Clasificador de Perros vs Gatos")
st.write("Sube una imagen para clasificar si es un perro o un gato")

IMG_SIZE = (224, 224)
MODEL_PATH = Path("modelo_perros_gatos.h5")

@st.cache_resource
def cargar_modelo():
    if MODEL_PATH.exists():
        return tf.keras.models.load_model(MODEL_PATH, compile=False)
    st.error("No se encontró el modelo. Asegúrate de tener 'modelo_perros_gatos.h5' en la misma carpeta.")
    st.stop()

def preparar_imagen(img):
    img = img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

def predecir(img):
    preds = modelo.predict(preparar_imagen(img), verbose=0)[0]
    clase_idx = np.argmax(preds)
    confianza = float(preds[clase_idx]) * 100
    clases = ['Gato', 'Perro']
    return clases[clase_idx], confianza

modelo = cargar_modelo()

archivo = st.file_uploader("Seleccione una imagen", type=["jpg", "jpeg", "png"])

if archivo:
    imagen = Image.open(archivo)
    st.image(imagen, caption="Imagen analizada", use_container_width=True)
    
    clase, confianza = predecir(imagen)
    
    st.subheader("Resultado")
    if "Perro" in clase:
        st.success(f"**{clase}** con {confianza:.2f}% de confianza")
    else:
        st.success(f"**{clase}** con {confianza:.2f}% de confianza")
    
    # Barra de progreso visual
    st.progress(int(confianza))
    st.write(f"**Confianza:** {confianza:.2f}%")
else:
    st.info("Cargue una imagen para iniciar la clasificación.")