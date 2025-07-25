import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# App title
st.title("Car vs Bike Image Classifier")

@st.cache_resource
def load_trained_model():
    return load_model("carbike-class2/model2.keras")
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



# Second model and uploader (no conflict with the first)
@st.cache_resource
def load_trained_model2():
    return load_model("carbike-class2/model5.keras")
model2 = load_trained_model2()

st.header("Car vs Bike Image Classifier (Model 2)")
uploaded_file2 = st.file_uploader("Choose an image for Model 2...", type=["jpg", "jpeg", "png"], key="uploader2")
if uploaded_file2:
    img2 = Image.open(uploaded_file2).resize((128, 128))
    st.image(img2, caption='Uploaded Image for Model 2', use_container_width=True)

    # Preprocess image
    img_array2 = np.array(img2) / 255.0
    img_array2 = np.expand_dims(img_array2, axis=0)

    # Predict and display result
    prediction2 = model2.predict(img_array2)
    # Handle both binary (sigmoid) and multi-class (softmax) outputs
    if prediction2.shape[-1] == 1:
        pred_value2 = prediction2[0][0]
        is_car2 = pred_value2 > 0.5
        label2 = class_names[int(is_car2)]
    else:
        pred_value2 = np.argmax(prediction2[0])
        label2 = class_names[pred_value2]
    st.write(f"Prediction (Model 2): **{label2}**")
