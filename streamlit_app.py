

#writefile score.py 
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import requests
from io import BytesIO

st.set_option('deprecation.showfileUploaderEncoding', False)



st.markdown(""" <style> .font {
font-size:50px ; font-family: 'Cooper Black'; color: #FF9633;} 
</style> """, unsafe_allow_html=True)
st.markdown('<p class="font">Brain Tumor Detection System</p>', unsafe_allow_html=True)

st.header('MADE BY:')
st.text('Ritik Dutt Sharma,Vibha Jaiswal, Km Manu, Devanshu Panwar, Harshit')
st.text('Rajkiyaengineeringcollege,kannauj')
st.text("Please Upload MRI image of Brain tumor For classification")

import base64
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"jpg"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('/content/drive/MyDrive/brain tumor detection/dataset_1/no/No12.jpg') 

@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('/content/drive/MyDrive/Colab Notebooks/braimtumor10epochs.h57')
    return model

with st.spinner('Loading Model into Memory....'):
    model = load_model()

classes=['Healthy_Brain','Tumor_Brain']

def scale(image):
    image = tf.cast(image, tf.float32)
    image /= 255.0

    return tf.image.resize(image,[244,244])

def decode_img(image):
    img = tf.image.decode_jpeg(image, channels=3)
    img = scale(img)
    return np.expand_dims(img, axis=0)

file = st.file_uploader("Please choose a file", type=['jpg'])

if file is None:
    st.text("Please upload an MRI image file in jpg/jpeg format")

else:
    content = file.getvalue()
    st.write("Pridicted Class :")
    with st.spinner('classifying....'):
        label = np.argmax(model.predict(decode_img(content)),axis=1)
        st.write(classes[label[0]])
    st.write("")
    image = Image.open(BytesIO(content))
    st.image(image, caption='Classifying MRI Image', use_column_width=True,output_format="auto")
    
    if __name__ == '__main__':
        main()
