import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

import sys
print(sys.path)  # Check paths
import tensorflow as tf
print(tf.__version__)  # Confirm TensorFlow is accessible


st.title("Car vs Bike Image Classifier")

# Load trained model
@st.cache_resource
def load_trained_model():
    model = load_model("model.h5")  # Update with correct path if necessary
    return model

model = load_trained_model()

uploaded_file = st.file_uploader("Choose an image...", type=s["jpg", "png", "jpeg"])
if uploaded_file is not None:
    img = Image.open(uploaded_file).resize((150, 150))
    st.image(img, caption='Uploaded Image', use_column_width=True)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)
    class_names = ["Bike", "Car"]  # Adjust if your labels are different
    st.write(f"Prediction: **{class_names[int(prediction[0][0] > 0.5)]}**")
