import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

st.title("Brain Tumor MRI Classification")

model = tf.keras.models.load_model("models/transfer_learning_tumournet_final.h5")
class_labels = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

uploaded_file = st.file_uploader("Upload an MRI image", type=["jpg","jpeg","png"])
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded MRI", use_column_width=True)
    img = np.array(image)
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    img_resized = cv2.resize(img, (224,224))/255.0
    img_resized = np.expand_dims(img_resized, axis=0)
    predictions = model.predict(img_resized)
    pred_class = class_labels[np.argmax(predictions)]
    confidence = np.max(predictions)
    st.success(f"Predicted Tumor Type: **{pred_class}**")
    st.info(f"Confidence: {confidence*100:.2f}%")
