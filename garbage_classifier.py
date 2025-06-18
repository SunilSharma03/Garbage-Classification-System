import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

model = load_model("model/garbage_model.h5")
labels = ['Cardboard', 'Glass', 'Metal', 'Paper', 'Plastic', 'Organic']

st.title("Garbage Classification System")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = load_img(uploaded_file, target_size=(150, 150))
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)/255.0

    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    class_label = labels[class_index]

    st.subheader(f"Prediction: {class_label}")
