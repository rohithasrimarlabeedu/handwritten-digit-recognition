import streamlit as st
import numpy as np
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas
from PIL import Image


# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Handwritten Digit Recognition",
    page_icon="🔢",
    layout="centered"
)


# -----------------------------
# Load Trained Model
# -----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "model/mnist_digit_model.keras"
    )


model = load_model()


# -----------------------------
# Title
# -----------------------------
st.title("🔢 Handwritten Digit Recognition")

st.write(
    "Draw a digit from 0 to 9 in the box below "
    "and click Predict Digit."
)


# -----------------------------
# Drawing Canvas
# -----------------------------
canvas_result = st_canvas(
    fill_color="black",
    stroke_width=12,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
    return_image_data=True
)


# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Digit"):

    if canvas_result.image_data is not None:

        # Get image from canvas
        image = canvas_result.image_data

        # Convert RGBA to grayscale
        image = Image.fromarray(
            image.astype("uint8"),
            "RGBA"
        ).convert("L")

        # Resize image to MNIST size
        image = image.resize((28, 28))

        # Convert image to NumPy array
        image_array = np.array(image).astype("float32")

        # Normalize pixel values
        image_array = image_array / 255.0

        # Add batch and channel dimensions
        image_array = image_array.reshape(
            1, 28, 28, 1
        )

        # Make prediction
        prediction = model.predict(
            image_array,
            verbose=0
        )

        # Get predicted digit
        predicted_digit = np.argmax(
            prediction[0]
        )

        # Get confidence
        confidence = (
            np.max(prediction[0]) * 100
        )

        # Display result
        st.success(
            f"Predicted Digit: {predicted_digit}"
        )

        st.info(
            f"Confidence: {confidence:.2f}%"
        )

        # -----------------------------
        # Prediction Probabilities
        # -----------------------------
        st.subheader("Prediction Probabilities")

        for digit, probability in enumerate(
            prediction[0]
        ):
            st.write(
                f"**{digit}:** "
                f"{probability * 100:.2f}%"
            )