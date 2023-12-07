'''
Setup instructions:
1. Import/Install all the required libraries from Step1 (if they are highlighted with red colour, they should
still work after they are installed)
2. Run the program
Szymon Kuczyński s22466
'''

import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
import matplotlib.pyplot as plt

# Load Fashion-MNIST dataset
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Normalize pixel values to between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# Define Model 1: Simple feedforward neural network
model1 = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile Model 1
model1.compile(optimizer='adam',
               loss='sparse_categorical_crossentropy',
               metrics=['accuracy'])

# Train Model 1
history1 = model1.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

# Define Model 2: Convolutional Neural Network (CNN)
model2 = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# Reshape the data for CNN model
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)

# Compile Model 2
model2.compile(optimizer='adam',
               loss='sparse_categorical_crossentropy',
               metrics=['accuracy'])

# Train Model 2
history2 = model2.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

# Evaluate Model 1
test_loss1, test_accuracy1 = model1.evaluate(test_images, test_labels)

# Evaluate Model 2
test_loss2, test_accuracy2 = model2.evaluate(test_images, test_labels)

# Compare metrics
print("Model 1 Accuracy:", test_accuracy1)
print("Model 2 Accuracy:", test_accuracy2)