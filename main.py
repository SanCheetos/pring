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

from myModel import check_faces_similarity

st.title("Модель для сравнивания 2-х лиц")

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
            st.text(type(uploaded_files1))
    
    with col4:
        if (uploaded_files2):
            st.header("Изображение 2")
            st.image(uploaded_files2)
    
    if (uploaded_files1 and uploaded_files2):
        st.text(f"distance: {check_faces_similarity(uploaded_files1.name, uploaded_files2.name)}")

# app = FastAPI()
# @app.post("/files/")
# async def create_file(file1: Annotated[UploadFile, File(...)], file2: Annotated[UploadFile, File(...)]):
#     uFile1 = await file1.read() 
#     uFile2 = await file2.read() 
#     return {"distance": check_faces_similarity(BytesIO(uFile1), BytesIO(uFile2))}

# @app.get("/")
# async def root():
#     return {"message": "Hello World"}
# from tests import test_models
# model_tests.similarImg()
# model_tests.noImage()
