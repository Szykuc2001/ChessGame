'''
Setup instructions:
1. Import/Install all the required libraries from Step1 (if they are highlighted with red colour, they should
still work after they are installed)
2. Run the program
Szymon Kuczyński s22466
'''

# Step 1: Import Libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Step 2: Load and Prepare Data
data = pd.read_csv('dataset1.csv')  # Replace 'diabetes_data.csv' with your dataset file path
X = data.drop('Outcome', axis=1)
y = data['Outcome']
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Step 3: Create Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Build the Neural Network Model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(12, input_dim=X.shape[1], activation='relu'),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Step 5: Compile the Model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Step 6: Train the Model
model.fit(X_train, y_train, epochs=150, batch_size=10)

# Step 7: Evaluate the Model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")