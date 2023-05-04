import cv2
import numpy as np
import streamlit as st

uploaded_file = st.file_uploader("Choose a image file", type=["jpg","png"])

if uploaded_file is not None:
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)

    # Display the image
    st.image(opencv_image, channels="BGR")
