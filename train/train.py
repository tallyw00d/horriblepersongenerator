from argparse import ArgumentParser
import os
import numpy as np
import pickle
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.callbacks import ModelCheckpoint

parser = ArgumentParser()
parser.add_argument("-i", "--in", dest="input",
                    help="location of input dataset")
parser.add_argument("--modelPath", "--out",dest="output",
                    help="location of model"
                    )

text = parser.parse_args().input
model_dir = parser.parse_args().output
# model_dir = os.path.abspath(os.environ.get('PS_MODEL_PATH'))

unique_chars = ''.join(sorted(set(text)))

n_unique_chars = len(unique_chars)
n_chars = len(text)

# dictionary that converts characters to integers
char2int = {c: i for i, c in enumerate(unique_chars)}
# dictionary that converts integers to characters
int2char = {i: c for i, c in enumerate(unique_chars)}

pickle.dump(char2int, open(model_dir+"char2int.pickle", "wb"))
pickle.dump(int2char, open(model_dir+"int2char.pickle", "wb"))

# hyper parameters
sequence_length = 100
step = 1
batch_size = 256
epochs = 80
sentences = []
y_train = []
for i in range(0, len(text) - sequence_length, step):
    sentences.append(text[i: i + sequence_length])
    y_train.append(text[i+sequence_length])

print("Number of sentences:", len(sentences))

# vectorization
X = np.zeros((len(sentences), sequence_length, n_unique_chars))
y = np.zeros((len(sentences), n_unique_chars))

for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char2int[char]] = 1
        y[i, char2int[y_train[i]]] = 1
print("X.shape:", X.shape)
print("y.shape:", y.shape)

## building the model

model = Sequential([
    LSTM(128, input_shape=(sequence_length, n_unique_chars)),
    Dense(n_unique_chars, activation="softmax"),
])

model.summary()
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

checkpoint = ModelCheckpoint(model_dir+"wonderland-v1-{loss:.2f}.h5", verbose=1)
model.fit(X, y, batch_size=batch_size, epochs=epochs, callbacks=[checkpoint])
