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

'''Load Fashion-MNIST Dataset: Loads the Fashion-MNIST dataset using fashion_mnist.load_data() and separates 
it into training and testing sets: train_images, train_labels, test_images, and test_labels.'''
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

'''Normalize Pixel Values: Scales pixel values in the images to a range between 0 and 1 by dividing train_images 
and test_images by 255.0.'''
train_images, test_images = train_images / 255.0, test_images / 255.0

'''Define Model 1: Creates a simple feedforward neural network using Keras Sequential API.

Reshapes the input images into a 1D array using Flatten layer to be compatible with a fully connected network.
Adds a dense layer with 128 neurons using ReLU activation.
Output layer with 10 neurons (equal to the number of classes in Fashion-MNIST) using the softmax activation for 
multi-class classification.'''
model1 = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

'''Compile Model 1: Configures the model for training by specifying the optimizer, loss function, and evaluation 
metrics similar to the previous example.'''
model1.compile(optimizer='adam',
               loss='sparse_categorical_crossentropy',
               metrics=['accuracy'])

'''Train Model 1: Fits the model to the training data for 10 epochs, validating the model's performance using the 
test data during training.'''
history1 = model1.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

'''Define Model 2: Constructs a CNN architecture using Keras Sequential API.

Starts with a convolutional layer (Conv2D) with 32 filters of size (3, 3) using ReLU activation and input shape of 
(28, 28, 1) (height, width, channels).
Adds a max-pooling layer (MaxPooling2D) with a pool size of (2, 2).
Flattens the output from convolutional layers.
Adds a dense layer with 128 neurons using ReLU activation.
Output layer with 10 neurons using softmax activation for classification.'''
model2 = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

'''Reshape Data for CNN Model: Reshapes train_images and test_images to match the input shape required for the CNN model.'''
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)

'''Compile Model 2: Configures the CNN model for training similarly to Model 1.'''
model2.compile(optimizer='adam',
               loss='sparse_categorical_crossentropy',
               metrics=['accuracy'])

'''Train Model 2: Fits the CNN model to the training data for 10 epochs, validating performance using test data.'''
history2 = model2.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

'''Evaluate Model 1 and Model 2: Computes the loss and accuracy of both models on the test dataset using 
model.evaluate() for both Model 1 and Model 2.'''
test_loss1, test_accuracy1 = model1.evaluate(test_images, test_labels)
test_loss2, test_accuracy2 = model2.evaluate(test_images, test_labels)

'''Print Model Metrics: Outputs the accuracy of both Model 1 and Model 2 to compare their performance on the test dataset.'''
print("Model 1 Accuracy:", test_accuracy1)
print("Model 2 Accuracy:", test_accuracy2)