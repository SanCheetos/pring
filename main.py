import kagglehub
import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
#from typing import Annotated
#from fastapi import FastAPI, File, UploadFile
#from io import BytesIO

path = kagglehub.model_download("faiqueali/facenet-tensorflow/tensorFlow2/default")
st.title("Модель для сравнивания 2-х лиц")
# Path to the saved model directory
model_dir = path

# Load the model
model = tf.saved_model.load(model_dir)

# Get the callable function from the loaded model
infer = model.signatures['serving_default']

def resize_image(image_path):
    img = Image.open(image_path)
    # изменяем размер
    new_image = img.resize((160, 160))
    st.image(new_image)
    new_image.save(f"/resize/temp")
    return f"/resize/temp"

def preprocess_image(img_path):
    """Load and preprocess the image."""
    img = image.load_img(img_path, target_size=(160, 160))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def get_face_embedding(img_path):
    """Generate face embedding from an image."""
    img = preprocess_image(img_path)
    # Perform inference using the callable function
    result = infer(tf.convert_to_tensor(img, dtype=tf.float32))
    embedding = result['Bottleneck_BatchNorm'].numpy()  # Use the correct output key
    return embedding

def check_faces_similarity(img_path1, img_path2, threshold=0.6):
    """Verify if two faces are the same person based on embeddings."""
    #img_pathR1 = resize_image(img_path1)
    #img_pathR2 = resize_image(img_path2)
    embedding1 = get_face_embedding(img_path1)
    embedding2 = get_face_embedding(img_path2)

    # Compute Euclidean distance between embeddings
    distance = np.linalg.norm(embedding1 - embedding2)
    distance = round(float(distance), 2)
    return distance
    
col1, col2 = st.columns(2)

with col1:
    st.header("Изображение 1")
    uploaded_files1 = st.file_uploader("", key="img1")

with col2:
    st.header("Изображение 2")
    uploaded_files2 = st.file_uploader("", key="img2")

if (uploaded_files1 or uploaded_files2):
    st.header("Превью")
    col3, col4 = st.columns(2)
    with col3:
        if (uploaded_files1):
            st.header("Изображение 1")
            st.image(uploaded_files1)
    
    with col4:
        if (uploaded_files2):
            st.header("Изображение 2")
            st.image(uploaded_files2)
    
    if (uploaded_files1 and uploaded_files2):
        st.text(f"distance: {check_faces_similarity(uploaded_files1, uploaded_files2)}")

# app = FastAPI()
# @app.post("/files/")
# async def create_file(file1: Annotated[UploadFile, File(...)], file2: Annotated[UploadFile, File(...)]):
#     uFile1 = await file1.read() 
#     uFile2 = await file2.read() 
#     return {"distance": check_faces_similarity(BytesIO(uFile1), BytesIO(uFile2))}

# @app.get("/")
# async def root():
#     return {"message": "Hello World"}
