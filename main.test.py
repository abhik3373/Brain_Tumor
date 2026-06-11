import cv2
from keras.models import load_model
from PIL import Image
import numpy as np




model = load_model('my_model_categorical.h5')

image = cv2.imread(r'D:\Brain Tumor Detector\MRI Scans\no\5 no.jpg') 
image = cv2.imread(r'D:\Brain Tumor Detector\MRI Scans\yes\Y3.jpg')  

img = Image.fromarray(image)
img = img.resize((64, 64))
img_array = np.array(img)

input_img = np.expand_dims(img_array, axis=0)
probabilities = model.predict(input_img)
predicted_class = np.argmax(probabilities)

print("Predicted class:", predicted_class)



