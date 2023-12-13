'''
Link to dataset used in the task: https://www.cs.toronto.edu/~kriz/cifar.html

Setup instructions:
1. Import/Install all the required libraries from Step1 (if they are highlighted with red colour, they should
still work after they are installed)
2. Run the program
Szymon Kuczyński s22466
'''

import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

'''Load CIFAR-10 Dataset: Loads the CIFAR-10 dataset using cifar10.load_data() function and assigns training and 
testing images along with their respective labels to variables train_images, train_labels, test_images, and test_labels.'''
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

'''Normalize Pixel Values: Scales pixel values in the images to a range between 0 and 1 by dividing train_images 
and test_images by 255.0.'''
train_images, test_images = train_images / 255.0, test_images / 255.0

'''Initialize Sequential Model: Creates a sequential model using Keras, which allows linear stacking of layers.
Add Convolutional Layers: Adds convolutional layers (Conv2D) with specific parameters:
First convolutional layer: 32 filters of size (3, 3) using ReLU activation, accepting input images of shape (32, 32, 3) 
(height, width, channels).
First max-pooling layer (MaxPooling2D) with a pool size of (2, 2).
Second convolutional layer: 64 filters of size (3, 3) using ReLU activation.
Second max-pooling layer with a pool size of (2, 2).
Third convolutional layer: 64 filters of size (3, 3) using ReLU activation.
Flatten and Dense Layers: Flattens the output from the convolutional layers into a 1D array and adds dense layers 
(Dense) with specific activations:
Dense layer with 64 neurons using ReLU activation.
Dropout layer (Dropout) with a dropout rate of 0.5 to prevent overfitting.
Output dense layer with 10 neurons using the softmax activation for multi-class classification (CIFAR-10 has 10 classes).'''
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

'''Compile the Model: Configures the model for training by specifying the optimizer, loss function, and evaluation metrics:
Optimizer: 'adam', an efficient gradient descent optimization algorithm.
Loss function: 'sparse_categorical_crossentropy' suitable for multi-class classification tasks.
Metrics to track during training: 'accuracy'.'''
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

'''Train the Model: Fits the model to the training data (train_images, train_labels) for 10 epochs. 
Also validates the model's performance using the test data during training (validation_data=(test_images, test_labels)).'''
history = model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

'''Evaluate the Model: Calculates the loss and accuracy of the trained model on the test dataset using model.evaluate().'''
test_loss, test_accuracy = model.evaluate(test_images, test_labels)

'''Calculate Predictions: Generates predictions using the trained model on the test images and obtains the class 
labels with the highest probability using np.argmax() along the last axis.'''
predictions = np.argmax(model.predict(test_images), axis=-1)

'''Generate Confusion Matrix: Computes the confusion matrix using confusion_matrix from Scikit-learn, 
comparing test_labels with predictions.'''
conf_matrix = confusion_matrix(test_labels, predictions)

'''Plot Confusion Matrix: Displays the confusion matrix using plt.imshow() to visualize the model's performance in 
classifying different categories.'''
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

'''Print Evaluation Metrics: Outputs the test accuracy obtained from model.evaluate().'''
print(f'Test accuracy: {test_accuracy}')