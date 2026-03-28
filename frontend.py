import streamlit as st
import requests
import numpy as np
from PIL import Image, ImageOps

st.set_page_config(page_title="MNIST Digit Predictor", page_icon="🔢")
st.title(" MNIST Digit Predictor")
st.write("Upload a handwritten digit image and the AI will predict what number it is!")

API_URL = "http://127.0.0.1:8000/predict"

uploaded_file = st.file_uploader("Upload a digit image (PNG or JPG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    st.image(image, caption="Your uploaded image", width=150)

    image = ImageOps.grayscale(image)
    image = image.resize((28, 28))
    img_array = np.array(image)
    img_list = img_array.flatten().tolist()

    if st.button(" Predict Digit"):
        with st.spinner("Asking the model..."):
            try:
                response = requests.post(API_URL, json={"data": img_list})

                if response.status_code == 200:
                    result = response.json()

                    # Works whether your FastAPI returns {"Digit": 5} or {"Digit:": 5}
                    if isinstance(result, dict):
                        digit = result.get("Digit") or result.get("Digit:")
                        st.success(f"Predicted Digit: **{digit}**")
                        st.balloons()
                    else:
                        st.error(f" Unexpected response from API: {result}")
                        st.info(" Fix your main.py: change  return {{\"Digit:\",digit}}  to  return {{\"Digit\": digit}}")
                else:
                    st.error(f" API Error: {response.status_code} - {response.text}")

            except requests.exceptions.ConnectionError:
                st.error("🔌 Could not connect to FastAPI. Make sure uvicorn is running on port 8000!")