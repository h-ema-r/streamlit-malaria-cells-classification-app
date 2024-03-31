import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2

# Load the trained model
model = tf.keras.models.load_model('models1/malaria-detection-model.h5')

# Function to preprocess the image
def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Function to get heatmap
def get_heatmap(image):
    img_array = preprocess_image(image)
    last_conv_layer = model.layers[-5].output  # Get the output of the last convolutional layer
    heatmap_model = tf.keras.models.Model(inputs=model.inputs, outputs=last_conv_layer)
    heatmap = heatmap_model.predict(img_array)[0]
    heatmap = np.mean(heatmap, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    return heatmap

# Function to overlay heatmap on the image
def overlay_heatmap(image, heatmap):
    # Resize heatmap to match image dimensions
    heatmap = cv2.resize(heatmap, (image.width, image.height))
    # Create a binary mask where heatmap values are above a certain threshold
    threshold = 0.5  # Adjust threshold as needed
    heatmap_binary = (heatmap > threshold).astype(np.uint8)
    # Apply the binary mask to the original image
    overlaid_image = np.copy(np.array(image))
    overlaid_image[heatmap_binary == 1] = [255, 0, 0]  # Highlight areas in red
    return overlaid_image


# Streamlit app
def main():
    st.title("Malaria Cell Classification App")
    st.write("Upload an image to classify and visualize the hotspots.")

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)

            # Predict
            if st.button('Classify'):
                processed_image = preprocess_image(image)
                prediction = model.predict(processed_image)[0]
                if prediction[0] > 0.2:
                    st.write("Prediction: Infected")
                else:
                    st.write("Prediction: Uninfected")

                # Get heatmap
                heatmap = get_heatmap(image)
                # Overlay heatmap on the original image
                overlay = overlay_heatmap(image, heatmap)
                st.image(overlay, caption='Heatmap Overlay', use_column_width=True)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == '__main__':
    main()
