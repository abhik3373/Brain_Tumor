import numpy as np
from PIL import Image
import cv2
import os
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

image_directory = 'MRI Scans/'

no_tumor_images = [f for f in os.listdir(image_directory + 'no/') if f.endswith('.jpg')]
yes_tumor_images = [f for f in os.listdir(image_directory + 'yes/') if f.endswith('.jpg')]
dataset = []
label = []

INPUT_SIZE = 64

for image_name in no_tumor_images:
    image = cv2.imread(image_directory + 'no/' + image_name)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    image = image.resize((INPUT_SIZE, INPUT_SIZE))
    dataset.append(np.array(image))
    label.append(0)

for image_name in yes_tumor_images:
    image = cv2.imread(image_directory + 'yes/' + image_name)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    image = image.resize((INPUT_SIZE, INPUT_SIZE))
    dataset.append(np.array(image))
    label.append(1)

dataset = np.array(dataset) / 255.0
label = np.array(label)

x_train, x_test, y_train, y_test = train_test_split(dataset, label, test_size=0.2, random_state=0)

# Model Building
model = Sequential()
model.add(Conv2D(32, (3, 3), kernel_initializer='he_uniform', input_shape=(INPUT_SIZE, INPUT_SIZE, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

# Binary CrossEntropy loss for binary classification
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(x_train, y_train, batch_size=16, epochs=20, validation_split=0.2, shuffle=True)

# Evaluate on test data
y_pred_prob = model.predict(x_test)
y_pred = (y_pred_prob > 0.5).astype(int)  

# Create confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Calculate accuracy from confusion matrix
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, 
            xticklabels=['No Tumor', 'Tumor'], yticklabels=['No Tumor', 'Tumor'])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()
