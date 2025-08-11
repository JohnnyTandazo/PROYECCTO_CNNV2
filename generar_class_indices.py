import json
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Ruta a la carpeta donde están organizadas tus clases (subcarpetas)
dataset_dir = 'DATASET ORDENADO'

# Asegurarse de que la carpeta existe
if not os.path.isdir(dataset_dir):
    print(f"ERROR: La ruta {dataset_dir} no existe.")
    exit()

# Crear un generador para obtener los índices de clases
datagen = ImageDataGenerator(rescale=1./255)
generator = datagen.flow_from_directory(
    dataset_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Imprimir las clases detectadas
print("Clases detectadas:")
print(generator.class_indices)

# Guardar el diccionario en un archivo JSON
with open('class_indices.json', 'w') as f:
    json.dump(generator.class_indices, f)

print("\nArchivo 'class_indices.json' generado correctamente.")
