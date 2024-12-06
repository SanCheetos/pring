import kagglehub
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import os.path


# from typing import Anno
def preprocess_image(img_path):
    """Load and preprocess the image."""
    img = image.load_img(img_path, target_size=(160, 160))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array


def get_face_embedding(img_path, infer):
    """Generate face embedding from an image."""
    img = preprocess_image(img_path)
    # Perform inference using the callable function
    result = infer(tf.convert_to_tensor(img, dtype=tf.float32))
    embedding = result['Bottleneck_BatchNorm'].numpy()
    # Use the correct output key
    return embedding


def check_file(file):
    if isinstance(str, type(file)):
        if not (os.path.exists(file)):
            return False
        else:
            return True
    else:
        return True


def check_faces_similarity(img_path1, img_path2, threshold=0.6):
    path = kagglehub.model_download(
        "faiqueali/facenet-tensorflow/tensorFlow2/default"
    )
    # Path to the saved model directory
    model_dir = path

    # Load the model
    model = tf.saved_model.load(model_dir)

    # Get the callable function from the loaded model
    infer = model.signatures['serving_default']
    """Verify if two faces are the same person based on embeddings."""
    # img_pathR1 = resize_image(img_path1)
    # img_pathR2 = resize_image(img_path2)
    check_files_result = [check_file(img_path1), check_file(img_path2)]
    if (check_files_result[0]) and (check_files_result[1]):
        embedding1 = get_face_embedding(img_path1, infer)
        embedding2 = get_face_embedding(img_path2, infer)
        # Compute Euclidean distance between embeddings
        distance = np.linalg.norm(embedding1 - embedding2)
        distance = round(float(distance), 2)
        return distance
    else:
        return "Один из файлов не существует"
