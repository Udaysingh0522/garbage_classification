import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from PIL import Image

# Load the trained model
model = load_model("garbage_classifier_model.h5")

# Define class names as per your dataset structure
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# Title
st.title("♻️ Garbage Classification Web App")
st.write("Upload a garbage image and see which category it belongs to!")

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = img.resize((150, 150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    # Show result
    st.success(f"Prediction: **{predicted_class}**")
    st.info(f"Confidence: **{confidence:.2f}%**")
