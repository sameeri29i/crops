import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2,preprocess_input as mobilenet_v2_preprocess_input
#from flask import Flask, render_template, Response, request
#import datetime, time
#import os, sys
#import numpy as np
#from threading import Thread

import base64

@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
   <style>
.stApp {
  background-image: url("data:image/png;base64,%s");
  background-size: cover;
}
</style>
    ''' % bin_str
    
    st.markdown(page_bg_img, unsafe_allow_html=True)
    return

set_png_as_page_bg('imageedit_3_7610295597.png')

LOGO_IMAGE = "iiee-removebg-preview.png"

st.markdown(
    """
    <style>
    .container {
        display: flex;
    }
    .logo-text {
        font-weight:700 !important;
        font-size:40px !important;
        color: #ffffff !important;
        padding-top: 0px !important;
    }
    .logo-img {
        float:left;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    f"""
    <div class="container">
        <img class="logo-img" src="data:image/png;base64,{base64.b64encode(open(LOGO_IMAGE, "rb").read()).decode()}">
        <p class="logo-text"> <br> ㅤInstitute Of Industrial Electronics Engineering</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.title("ㅤㅤㅤ ㅤㅤIoT Based Crop Monitoring and Analysis")
model = tf.keras.models.load_model("fypmodel3.h5")
#faceCascade = cv2.CascadeClassifier(cv2.haarcascades+'haarcascade_frontalface_default.xml')
### load file
uploaded_file = st.file_uploader("Choose a image file")


map_dict = {0: 'Apple scab',
            1: 'Apple Black rot',
            2: 'Apple Cedar apple rust',
            3: 'Apple healthy',
            4: 'Blueberry healthy',
            5: 'Apple health',
            6: 'Apple Healthy',
            7: 'Corn (maize) Northern Leaf Blight',
            8: 'Corn (maize) Common rust ',
            9: 'Corn (maize) Northern Leaf Blight',
            10: 'Corn (maize) (Cercospora leaf spot',
            11: 'Grape Black rot',
            12: 'Grape Esca (Black Measles)',
            13: 'Grape Leaf blight (Isariopsis Leaf Spot)',
            14: 'Apple healthy',
            15: 'Apple healthy',
            16: 'Peach Bacterial spot',
            17: 'Tomato healthy',
            18: 'Pepper, bell Bacterial spot',
            19: 'Pepper, bell healthy',
            20: 'Apple Healthy',
            21: 'Apple Healthy',
            22: 'Tomato healthy',
            23: 'Raspberry healthy',
            24: 'Soybean healthy',
            25: 'Squash Powdery mildew',
            26: 'Strawberry Leaf scorch',
            27: 'Strawberry healthy',
            28: 'Apple Healthy',
            29: 'Tomato Early blight',
            30: 'Tomato Late blight',
            31: 'Tomato Leaf Mold',
            32: 'Tomato Septoria leaf spot',
            33: 'Tomato Spider mites',
            34: 'Tomato Target Spot',
            35: 'Tomato Yellow Leaf Curl Virus',
            36: 'Tomato mosaic virus',
            37: 'Tomato healthy'}


if uploaded_file is not None:
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(opencv_image,(224,224))
    # Now do something with the image! For example, let's display it:
    st.image(opencv_image, channels="RGB", use_column_width=True)

    resized = mobilenet_v2_preprocess_input(resized)
    img_reshape = resized[np.newaxis,...]

    Genrate_pred = st.button("Generate Prediction")    
    if Genrate_pred:
        prediction = model.predict(img_reshape).argmax()
        st.title("Predicted Label for the image is")
        st.title("{}".format(map_dict [prediction]))
