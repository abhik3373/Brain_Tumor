import numpy as np
import cv2
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.utils import to_categorical

# Define the path to the image directory
image_directory = 'MRI Scans/'

# Load images and labels
def load_images_and_labels(directory):
    images = []
    labels = []
    for category in os.listdir(directory):
        if category == 'no':
            label = 0  # 'no tumor'
        elif category == 'yes':
            label = 1  # 'tumor'
        else:
            continue
        
        category_path = os.path.join(directory, category)
        for filename in os.listdir(category_path):
            if filename.endswith('.jpg'):
                image_path = os.path.join(category_path, filename)
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image)
                image = image.resize((64, 64))  # Resize image to (64, 64)
                images.append(np.array(image))
                labels.append(label)
    
    return np.array(images), np.array(labels)

# Load dataset
dataset, label = load_images_and_labels(image_directory)

# Split data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(dataset, label, test_size=0.2, random_state=0)

# Normalize pixel values (optional but recommended)
x_train = x_train / 255.0
x_test = x_test / 255.0

# Model Building
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))  # Output layer for binary classification

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(x_train, y_train, batch_size=16, epochs=15, validation_split=0.2, shuffle=True)

# Evaluate the model on test data
loss, accuracy = model.evaluate(x_test, y_test)
print("Test Accuracy:", accuracy)