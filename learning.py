import streamlit as st
import cv2
from keras.models import load_model
from PIL import Image
import numpy as np

# Load the pre-trained model
model = load_model('my_model_categorical.h5')

def preprocess_image(image):
    # Convert BGR image to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Resize image to match model input size
    image = cv2.resize(image, (64, 64))
    return image

def predict_tumor(image):
    # Preprocess image
    img_array = preprocess_image(image)
    # Expand dimensions to create a batch of size 1
    input_img = np.expand_dims(img_array, axis=0)
    # Predict class probabilities
    probabilities = model.predict(input_img)
    predicted_class = np.argmax(probabilities)
    return predicted_class

def main():
    # Custom CSS for setting background color and text color
    st.markdown(
        """
        <style>
        .title {
            color: yellow;
        }
        .text {
            color: yellow;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Set page title and description with custom CSS classes
    st.markdown('<p class="title">BRAIN TUMOR DETECTOR</p>', unsafe_allow_html=True)
    st.markdown('<p class="text">Upload an MRI scan image to detect the presence of a brain tumor.</p>', unsafe_allow_html=True)

    # File uploader to upload an image
    uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg"])

    if uploaded_file is not None:
        # Read uploaded image
        image = Image.open(uploaded_file)
        img_array = np.array(image)

        # Display uploaded image
        st.image(image, caption='Uploaded MRI Image', use_column_width=True)

        # Check if prediction button is clicked
        if st.button("Predict"):
            # Predict tumor presence
            predicted_class = predict_tumor(img_array)

            # Display prediction result
            if predicted_class == 0:
                st.success("No tumor detected.")
            else:
                st.error("Tumor detected.")
    
if __name__ == '__main__':
    main()
