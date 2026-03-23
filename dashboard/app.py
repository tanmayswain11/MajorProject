import sys, os

# Fix module path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import streamlit as st
from model.cnn.predict import predict_image

st.title("Drone Detection")

file = st.file_uploader("Upload image")

if file:
    with open("temp.png", "wb") as f:
        f.write(file.read())

    result = predict_image("temp.png")
    st.write(result)