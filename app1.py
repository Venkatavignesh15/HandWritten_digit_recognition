import streamlit as st
import numpy as np
from PIL import Image
from keras.models import load_model
from streamlit_drawable_canvas import st_canvas

# Load model
model = load_model("mnist.h5")

st.title("✍️ Handwritten Digit Recognition")

# Canvas
canvas = st_canvas(
    fill_color="black",
    stroke_width=15,
    stroke_color="white",
    background_color="black",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas",
)

if st.button("Predict"):
    if canvas.image_data is not None:
        img = canvas.image_data[:, :, 0]
        img = Image.fromarray(img).resize((28, 28))
        img = np.array(img) / 255.0
        img = img.reshape(1, 28, 28, 1)

        pred = model.predict(img)[0]
        st.success(f"Prediction: {np.argmax(pred)}")
        st.write("Confidence:", np.max(pred))
