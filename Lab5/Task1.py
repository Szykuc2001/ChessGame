'''
Link to dataset used in the task: https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv

Setup instructions:
1. Import/Install all the required libraries from Step1 (if they are highlighted with red colour, they should
still work after they are installed)
2. Download dataset1.csv and put it in the same directory as Task1.py
3. Run the program
Szymon Kuczyński s22466
'''


import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

'''Loading Data: Reads a CSV file named 'dataset1.csv' using Pandas and stores it in the variable data.'''
data = pd.read_csv('dataset1.csv')
'''Splitting Features and Target: Separates the data into features (X) and the target variable (y). Here, 
Outcome is considered the target column, and the rest are features.'''
X = data.drop('Outcome', axis=1)
y = data['Outcome']
'''Scaling Features: Uses StandardScaler from Scikit-learn to standardize the feature values (mean=0, variance=1) in X'''
scaler = StandardScaler()
X = scaler.fit_transform(X)

'''Train-Test Split: Splits the dataset into training and testing sets using train_test_split from Scikit-learn. 
The parameter test_size=0.2 specifies that 20% of the data will be used for testing, while 80% will be used for training. 
random_state=42 sets a seed for reproducibility.'''
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

'''Initialize Neural Network: Creates a sequential model using Keras, a high-level neural networks API running on top of 
TensorFlow.
Add Layers: Adds three layers to the model:
Input layer (Dense with 12 neurons), matching the number of features in X. Activation function used is Rectified Linear 
Unit (ReLU).
Hidden layer (Dense with 8 neurons) also using ReLU activation.
Output layer (Dense with 1 neuron) using the Sigmoid activation function for binary classification tasks.'''
model = tf.keras.Sequential([
    tf.keras.layers.Dense(12, input_dim=X.shape[1], activation='relu'),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

'''Configure Model for Training: Configures the model for training by specifying the loss function, optimizer, and 
evaluation metric(s).
Loss function: 'binary_crossentropy' suitable for binary classification problems.
Optimizer: 'adam', an efficient gradient descent optimization algorithm.
Metrics to track during training: 'accuracy' to measure model performance.'''
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

'''Fit Model to Training Data: Trains the neural network model on the training data (X_train, y_train) for 150 epochs
 (passes through the entire dataset) with a batch size of 10.
The model iteratively adjusts its weights based on the optimization algorithm (adam) to minimize the specified loss function.'''
model.fit(X_train, y_train, epochs=150, batch_size=10)

'''Assess Model Performance: Evaluates the trained model's performance on the test data (X_test, y_test) 
by calculating the loss and accuracy.
Prints the test accuracy as a percentage.'''
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")