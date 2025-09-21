import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import base64

# ================= Load Model =================
@st.cache_resource
def load_trained_model():
    return load_model("model.h5")

model = load_trained_model()

# ✅ Class labels (must match training order)
class_labels = ["cats", "dogs", "elephants", "human", "Peacock", "pigs"]

# ================= Helper Functions =================
def add_bg_from_local(image_file):
    """Set background image from local file"""
    with open(image_file, "rb") as file:
        data = file.read()
    encoded = base64.b64encode(data).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

def predict_image(uploaded_file, threshold=0.6):
    """Predict class or return Unknown if confidence is too low"""
    img = Image.open(uploaded_file).convert("RGB")
    img_resized = img.resize((224, 224))   # ✅ Same size as training

    img_array = image.img_to_array(img_resized)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    confidence = np.max(prediction)

    if confidence < threshold:
        return "Unknown / Don't Know", confidence, img
    else:
        return class_labels[class_index], confidence, img

# ================= Streamlit UI =================
st.set_page_config(page_title="Animal Recognition", page_icon="🐾", layout="centered")
add_bg_from_local("assets/background.png")

st.markdown("<h1 style='text-align: center; color: white;'>🐾 Animal Recognition App</h1>", unsafe_allow_html=True)
st.write("Upload an image, and the model will recognize the animal or say **Unknown** if not sure.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    label, confidence, img = predict_image(uploaded_file)

    st.image(img, caption="Uploaded Image", use_container_width=True)  # ✅ updated

    if label == "Unknown / Don't Know":
        st.markdown(
            f"<h3 style='text-align: center; color: red;'>Prediction: {label} (Confidence {confidence*100:.2f}%)</h3>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<h3 style='text-align: center; color: yellow;'>Prediction: {label} ({confidence*100:.2f}% confidence)</h3>",
            unsafe_allow_html=True
        )
