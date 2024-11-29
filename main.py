import kagglehub
import streamlit as st
import tensorflow as tf
import numpy as np
import requests
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
from io import BytesIO
import os

# Функция для загрузки изображения по URL
def load_image_from_url(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return img

# Функция для изменения размера изображения
def resize_image(img):
    new_image = img.resize((160, 160))
    st.image(new_image)
    
    # Создаем директорию для сохранения изображений, если она не существует
    save_dir = "resized_images"
    os.makedirs(save_dir, exist_ok=True)
    
    save_path = os.path.join(save_dir, "resized_image.jpg")
    new_image.save(save_path)
    
    return save_path

# Функция для предварительной обработки изображения перед передачей в модель
def preprocess_image(img_path):
    """Load and preprocess the image."""
    img = image.load_img(img_path, target_size=(160, 160))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# Функция для получения эмбеддинга лица
def get_face_embedding(img_path):
    """Generate face embedding from an image."""
    img = preprocess_image(img_path)
    # Perform inference using the callable function
    result = infer(tf.convert_to_tensor(img, dtype=tf.float32))
    
    # Печать всех доступных ключей вывода модели
    print(result.keys())  # Для отладки
    
    embedding = result['Bottleneck_BatchNorm'].numpy()  # Убедитесь, что этот ключ существует
    return embedding

# Функция для проверки сходства лиц
def check_faces_similarity(img_url1, img_url2, threshold=0.6):
    """Verify if two faces are the same person based on embeddings."""
    # Загрузить изображения по URL
    img1 = load_image_from_url(img_url1)
    img2 = load_image_from_url(img_url2)
    
    # Изменить размер изображений
    img_path1 = resize_image(img1)
    img_path2 = resize_image(img2)
    
    embedding1 = get_face_embedding(img_path1)
    embedding2 = get_face_embedding(img_path2)

    # Вычислить евклидово расстояние между эмбеддингами
    distance = np.linalg.norm(embedding1 - embedding2)
    distance = round((distance * 10), 2)
    st.title(f'Разница лиц: {distance}%')

# Пример использования
url1 = 'https://github.com/SanCheetos/pring/blob/main/face1.jpg?raw=true'
url2 = 'https://github.com/SanCheetos/pring/blob/main/face2.jpg?raw=true'

# Отображаем изображения для предварительного просмотра
col1, col2 = st.columns(2)

with col1:
    st.header("Изображение 1")
    st.image(url1)

with col2:
    st.header("Изображение 2")
    st.image(url2)

# Проводим проверку сходства лиц
if url1 and url2:
    st.header("Результат")
    check_faces_similarity(url1, url2)

