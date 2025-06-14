import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# App title
st.title("Car vs Bike Image Classifier")

@st.cache_resource
def load_trained_model():
     # Load your trained model; adjust path if needed
     return load_model("carbike-class/carbike-class2/model1.keras")

model = load_trained_model()

# Image uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file:
    img = Image.open(uploaded_file).resize((150, 150))
    st.image(img, caption='Uploaded Image', use_container_width=True)

    # Preprocess image
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict and display result
    prediction = model.predict(img_array)[0][0]
    class_names = ["Bike", "Car"]
    label = class_names[int(prediction > 0.5)]
    st.write(f"Prediction: **{label}**")