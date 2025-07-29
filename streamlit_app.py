import streamlit as st
import os
import warnings
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

st.set_page_config(
    page_title="Cat vs Dog Classifier",
    page_icon="",
    layout="centered"
)

@st.cache_resource
def load_model_once():
    try:
        model_path = "model/catsdogs.h5"
        if not os.path.exists(model_path):
            st.error(f"Model file not found at {model_path}")
            return None
        
        model = load_model(model_path, compile=False)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def main():
    st.title("Cat vs Dog Classifier")
    st.write("Upload an image and I'll tell you if it's a cat or a dog!")
    
    model = load_model_once()
    
    if model is None:
        st.error("Model could not be loaded.")
        return
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        try:
            img = Image.open(uploaded_file).convert("RGB")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(img, caption="Uploaded Image", use_column_width=True)
            
            with col2:
                st.write("**Image Details:**")
                st.write(f"Name: {uploaded_file.name}")
                st.write(f"Size: {uploaded_file.size:,} bytes")
                st.write(f"Type: {uploaded_file.type}")
            
            if st.button("Classify Image", type="primary"):
                with st.spinner("Analyzing..."):
                    try:
                        img_resized = img.resize((150, 150))
                        img_array = image.img_to_array(img_resized) / 255.0
                        img_array = np.expand_dims(img_array, axis=0)
                        
                        prediction = model.predict(img_array, verbose=0)
                        confidence = float(prediction[0][0])
                        
                        if confidence > 0.5:
                            label = "🐶 Dog"
                            conf_percentage = confidence * 100
                        else:
                            label = "🐱 Cat"
                            conf_percentage = (1 - confidence) * 100
                        
                        st.success(f"**Prediction: {label}**")
                        st.info(f"**Confidence: {conf_percentage:.1f}%**")
                        st.progress(conf_percentage / 100)
                        
                    except Exception as e:
                        st.error(f"Error during prediction: {str(e)}")
            
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    
    else:
        st.info("Please upload an image to get started!")

if __name__ == "__main__":
    main()