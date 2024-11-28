import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Path to the saved model directory
model_dir = '/kaggle/input/facenet-tensorflow/tensorflow2/default/2/'

# Load the model
model = tf.saved_model.load(model_dir)

# Get the callable function from the loaded model
infer = model.signatures['serving_default']

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
    embedding1 = get_face_embedding(img_path1)
    embedding2 = get_face_embedding(img_path2)

    # Compute Euclidean distance between embeddings
    distance = np.linalg.norm(embedding1 - embedding2)
    print(f'Distance between faces: {distance}')

# Example usage
img_path1 = '/path/to/face1.jpg'
img_path2 = '/path/to/face2.jpg'

check_faces_similarity(img_path1, img_path2)
