import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from PIL import Image

# Load your trained model
model = load_model("Moodclassifier.h5")

# Define a function to preprocess the image
def preprocess_image(img):
    img = img.resize((256, 256))  # Resize to match model input size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize pixel values
    return img_array


# Streamlit UI
st.title("😊 Mood Classifier (Happy or Sad) 😢")
st.write("Upload an image to predict the person's mood.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display uploaded image
    image_pil = Image.open(uploaded_file)
    st.image(image_pil, caption="Uploaded Image", use_container_width=True)

    # Preprocess and predict
    img_array = preprocess_image(image_pil)
    prediction = model.predict(img_array)

    # Assuming binary classification (0 = Sad, 1 = Happy)
    mood = "Happy 😊" if prediction[0][0] > 0.5 else "Sad 😢"

    # Display result
    st.subheader("Prediction:")
    st.write(f"The person in the image looks **{mood}**.")

