import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import json

# Load the model
model = load_model('pixelpunk.h5')

# Load class mapping
with open('class_mapping.json', 'r') as f:
    class_mapping = json.load(f)
class_names = {v: k for k, v in class_mapping.items()}

img_height, img_width = 224, 224

# Function to preprocess the image
def preprocess_image(image):
    image = image.resize((img_width, img_height))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# Function to classify the image
def classify_image(image):
    image_array = preprocess_image(image)
    predictions = model.predict(image_array)
    class_id = np.argmax(predictions, axis=1)[0]
    return class_id

# Streamlit app
st.title('Weapon Detection App')
st.write('Upload an image to classify it.')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("Classifying...")

    class_id = classify_image(image)
    weapon_name = class_names.get(class_id, "Unknown")

    st.write(f'Predicted class label: {weapon_name}')