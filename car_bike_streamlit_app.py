import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# App title
st.title("Car vs Bike Image Classifier")

@st.cache_resource
def load_trained_model():
     # Load your trained model; adjust path if needed
     return load_model("carbike-class\carbike-class2\model3.keras")

model = load_trained_model()

# Define class names in the order used during model training
class_names = ["Bike", "Car"]

# Image uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file:
    img = Image.open(uploaded_file).resize((128, 128))
    st.image(img, caption='Uploaded Image', use_container_width=True)

    # Preprocess image
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict and display result
    prediction = model.predict(img_array)
    # Update class_names to match the label order used during model training
    # Handle both binary (sigmoid) and multi-class (softmax) outputs
    if prediction.shape[-1] == 1:
        pred_value = prediction[0][0]
        is_car = pred_value > 0.5  # True if model predicts 'Car', False for 'Bike'
        label = class_names[int(is_car)]
    else:
        pred_value = np.argmax(prediction[0])
        label = class_names[pred_value]
        label = class_names[pred_value]
    st.write(f"Prediction: **{label}**")