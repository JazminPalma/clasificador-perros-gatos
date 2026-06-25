#Aumento de datos
#Generar imagenes 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os

#Crear el dataset generador
datagen = ImageDataGenerator(
    rescale=1. / 255,              #Normalizarlas 
    rotation_range = 10,           #rotar un angulo de 10 grados (ambos lados)
    width_shift_range=0.15,
    height_shift_range = 0.15,
    shear_range = 5,               #deforme un poco
    zoom_range = [0.7, 1.3],       #se aleje en 0.7 o un zoom de 1,3
    validation_split = 0.2         #20% de las imagenes sean para las pruebas
)

#shuffle (mezclar constantemente)
data_gen_entrenamiento = datagen.flow_from_directory(
    "dataset/",                    # <--- SOLO CAMBIÉ ESTO (era /content/dataset)
    target_size=(224,224),
    batch_size=32, 
    shuffle=True,
    subset="training"
)

#20% de entrenamiento
data_gen_pruebas = datagen.flow_from_directory(
    "dataset/",                   
    target_size=(224,224),
    batch_size=32, 
    shuffle=True,
    subset="validation"
)

print(f"Entrenamiento: {data_gen_entrenamiento.samples}")
print(f"Pruebas: {data_gen_pruebas.samples}")

# ============================================
# NUEVO: USAR MOBILENETV2 (Como dice la presentación)
# ============================================

print("\nUsamos MobileNetV2 pre-entrenado.")

# Cargar MobileNetV2 sin la capa de clasificación
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)

# Congelar las capas (NO se entrenan)
base_model.trainable = False

# Construir el modelo completo
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(2, activation='softmax')  # 2 clases: perro, gato
])

# Compilar
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("Modelo compilado")
model.summary()

# ============================================
# NUEVO: ENTRENAR
# ============================================

print("\nEntrenando el modelo.")
print("=" * 50)

history = model.fit(
    data_gen_entrenamiento,
    validation_data=data_gen_pruebas,
    epochs=10,
    steps_per_epoch=data_gen_entrenamiento.samples // 32,
    validation_steps=data_gen_pruebas.samples // 32
)

# ============================================
# NUEVO: EVALUAR Y GUARDAR
# ============================================

print("\nEvaluar el modelo.")
loss, accuracy = model.evaluate(data_gen_pruebas, verbose=0)
print(f"Precisión en pruebas: {accuracy:.2%}")

# Guardar modelo
model.save('modelo_perros_gatos.h5')

# ============================================
# NUEVO: PROBAR CON IMÁGENES
# ============================================

print("\nProbar el modelo con imágenes del dataset.")
print("=" * 50)

from tensorflow.keras.preprocessing import image

def predecir_imagen(ruta_imagen):
    """Predice si la imagen es perro o gato"""
    img = image.load_img(ruta_imagen, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    prediccion = model.predict(img_array, verbose=0)
    clase_idx = np.argmax(prediccion[0])
    confianza = prediccion[0][clase_idx]
    
    clases = ['gato', 'perro']
    return clases[clase_idx], confianza

# Probar una imagen de perro
try:
    imagenes_perros = os.listdir("dataset/perros")
    if imagenes_perros:
        ruta = f"dataset/perros/{imagenes_perros[0]}"
        clase, confianza = predecir_imagen(ruta)
        print(f"Perro de prueba: {imagenes_perros[0]}")
        print(f"Predicción: {clase} ({confianza:.2%})")
except:
    print("No se encontraron imágenes de perros")

# Probar una imagen de gato
try:
    imagenes_gatos = os.listdir("dataset/gatos")
    if imagenes_gatos:
        ruta = f"dataset/gatos/{imagenes_gatos[0]}"
        clase, confianza = predecir_imagen(ruta)
        print(f"Gato de prueba: {imagenes_gatos[0]}")
        print(f"Predicción: {clase} ({confianza:.2%})")
except:
    print("No se encontraron imágenes de gatos")

# ============================================
# FINAL
# ============================================

print("\n" + "=" * 60)
print("=" * 60)
print(f"   - Dataset: {data_gen_entrenamiento.samples + data_gen_pruebas.samples} imágenes")
print(f"   - Precisión: {accuracy:.2%}")