'''
Link to dataset used in task: https://keras.io/api/datasets/imdb/

Setup instructions:
1. Import/Install all the required libraries from Step1 (if they are highlighted with red colour, they should
still work after they are installed)
2. Run the program
Szymon Kuczyński s22466
'''

import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, GlobalAveragePooling1D
from tensorflow.keras.preprocessing import sequence

'''Load IMDB Dataset: Loads the IMDB dataset using imdb.load_data() function and restricts the vocabulary to the most 
frequent 10,000 words (num_words=max_words). It separates the data into training and testing sets: train_data, 
train_labels, test_data, and test_labels.'''
max_words = 10000
maxlen = 200
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=max_words)

'''Sequence Padding: Applies padding (sequence.pad_sequences) to ensure all sequences have a consistent length 
(maxlen=200) by truncating longer sequences and padding shorter sequences with zeros.'''
train_data = sequence.pad_sequences(train_data, maxlen=maxlen)
test_data = sequence.pad_sequences(test_data, maxlen=maxlen)

'''Define LSTM-based Model: Constructs a sequential model using Keras.

Embedding layer (Embedding) creates word embeddings for each word index in the input sequences. It maps integers 
(word indices) to dense vectors of fixed size.
LSTM layer (LSTM) with 64 units, a type of recurrent neural network (RNN) well-suited for sequential data, processing 
sequences and capturing dependencies between words.
Dense layer (Dense) with a single neuron using the sigmoid activation function for binary classification 
(positive or negative sentiment).'''
model = Sequential([
    Embedding(max_words, 64, input_length=maxlen),
    LSTM(64),
    Dense(1, activation='sigmoid')
])

'''Compile the Model: Configures the model for training using the Adam optimizer, 
binary cross-entropy loss (suitable for binary classification), and tracking the accuracy metric.'''
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

'''Train the Model: Fits the model to the training data for 5 epochs with a batch size of 128 and a validation 
split of 20%.'''
history = model.fit(train_data, train_labels,
                    epochs=5,
                    batch_size=128,
                    validation_split=0.2)

'''Evaluate the Model: Computes the loss and accuracy of the trained model on the test dataset using model.evaluate().'''
test_loss, test_acc = model.evaluate(test_data, test_labels)
print(f"Test accuracy: {test_acc}")