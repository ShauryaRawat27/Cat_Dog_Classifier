import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

model = load_model("catsdogs.h5")

st.title("Dog vs Cat Classifier ")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).resize((150, 150))
    st.image(img, caption='Uploaded Image', use_column_width=True)
    
    if st.button("Classify"):
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        prediction = model.predict(img_array)[0][0]
        
        label = "Dog" if prediction > 0.5 else "Cat"
        st.subheader(f"Prediction: **{label}** ({prediction:.2f})")
