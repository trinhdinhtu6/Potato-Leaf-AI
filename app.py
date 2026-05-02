import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import streamlit as st
from PIL import Image
import numpy as np

st.set_page_config(layout="wide", page_title="Potato AI Assist", page_icon="🥔")


st.markdown("<h1 style='text-align: center; color: #2E7D32;'>Potato Leaves Diseases Detector</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Use Deep Learning (MobileNetV2) to recognize agricultural diseases early.</p>", unsafe_allow_html=True)
st.markdown("---")


@st.cache_resource
def load_my_model():
    
    model = tf.keras.models.load_model('potato_model.keras')
    return model

try:
    with st.spinner('Initiating with AI...'):
        model = load_my_model()
except Exception as e:
    st.error(f"Unable to find model file: {e}")
    st.stop()


class_names = ['Early Blight', 'Healthy', 'Late Blight']


st.subheader("📸 Upload leaves image to analyze...")
uploaded_file = st.file_uploader("Choose picture types (JPG, PNG)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        image = Image.open(uploaded_file)
        st.image(image, caption='Image uploaded', use_container_width=True)
        
    with col2:
        st.subheader("🔍 AI Diagnosis Result:")
       

        img = image.resize((224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        
       
        with st.spinner('AI is analyzing'):
            predictions = model.predict(img_array)
           
            score = predictions[0] 
            predicted_class = class_names[np.argmax(score)]
            confidence = np.max(score) * 100
            
         
            if predicted_class == 'Healthy':
                st.success(f"Result: **{predicted_class}**")
            else:
                st.error(f"Result: **{predicted_class}**")
                
            st.write(f"Confidence: **{confidence:.2f}%**")
            
          
            st.info("**Reminder:** Moisture and density of plants should be checked if there are signs of mold.")

st.markdown("---")
st.caption("Potato Leaves Diseases Detector | Student: Trịnh Đình Tú")
