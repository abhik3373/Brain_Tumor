import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Function to preprocess image for model prediction
def preprocess_image(image):
    # Resize image to 64x64 pixels and convert to RGB
    image = image.resize((64, 64))
    image_array = np.array(image)
    image_array = image_array.astype(np.float32) / 255.0  # Normalize pixel values
    return image_array

# Load pre-trained brain tumor detection model
@st.cache(allow_output_mutation=True)
def load_detection_model():
    model_path = 'my_model_categorical.keras'
    model = load_model(model_path)
    return model

# Streamlit app
def main():
    st.title("Brain Tumor Detection")

    uploaded_file = st.file_uploader("Upload a brain MRI image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded MRI Image', use_column_width=True)

        # Check if user clicked the 'Detect' button
        if st.button("Detect Tumor"):
            # Load the detection model
            detection_model = load_detection_model()

            # Preprocess the image
            processed_image = preprocess_image(image)

            # Reshape processed image to match the expected input shape of the model
            processed_image = np.expand_dims(processed_image, axis=0)  # Add batch dimension
            processed_image = processed_image.reshape(-1, 64, 64, 3)  # Reshape to (1, 64, 64, 3)

            # Perform prediction
            prediction = detection_model.predict(processed_image)

            # Display prediction result
            if prediction[0][0] > 0.5:
                st.error("Brain tumor detected!")
            else:
                st.success("No brain tumor detected.")

# Run the app
if __name__ == '__main__':
    main()
