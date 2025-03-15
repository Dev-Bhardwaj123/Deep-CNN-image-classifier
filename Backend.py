import streamlit as st
import pickle
import numpy as np
from PIL import Image
from sklearn.preprocessing import StandardScaler

# Load the pre-trained model
with open("mood_classifier.pkl", "rb") as f:
    model = pickle.load(f)


# Function to preprocess image
def preprocess_image(image):
    image = image.convert("L")  # Convert to grayscale
    image = image.resize((64, 64))  # Resize to match model input size
    img_array = np.array(image).flatten()  # Flatten the image
    img_array = img_array / 255.0  # Normalize
    return img_array.reshape(1, -1)


# Streamlit UI
st.title("Happy/Sad Image Classifier")
st.write("Upload an image and the model will classify it as Happy or Sad.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess image
    img_data = preprocess_image(image)

    # Make prediction
    prediction = model.predict(img_data)[0]

    # Display result
    result = "Happy 😊" if prediction == 1 else "Sad 😢"
    st.markdown(f"## Prediction: **{result}**")

