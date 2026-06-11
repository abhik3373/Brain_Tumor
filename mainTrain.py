import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Data Preparation
data_dir = "D:/Brain Tumor Detector/MRI Scans"  # Adjust if needed
categories = ['no', 'yes']  # Folder names for the two classes

dataset = []
label = []

for i, category in enumerate(categories):
    folder_path = os.path.join(data_dir, category)
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        try:
            img = cv2.imread(img_path)
            img = cv2.resize(img, (64, 64))
            dataset.append(img)
            label.append(i)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")

dataset = np.array(dataset, dtype='float32') / 255.0  # Normalize to [0,1]
label = np.array(label)

# One-hot encoding
label = to_categorical(label, num_classes=2)

# Train-Test Split
x_train, x_test, y_train, y_test = train_test_split(dataset, label, test_size=0.2, random_state=42)

# Model Building
model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

# Compilation
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Training
model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# Save the model (Keras 3 compatible format: .h5)
model.save("brain_tumor_model.h5")
