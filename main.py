import matplotlib.pyplot
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils.np_utils import normalize as nrm
import numpy as np


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Strg+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    mnist = keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = nrm(x_train, axis=1)
    x_test = nrm(x_test, axis=1)
    model = Sequential()
    model.add(Dense(128, activation='relu'))  # Hidden Layer 1
    model.add(Dense(128, activation='relu'))  # Hidden Layer 2
    model.add(Dense(10, activation='softmax'))  # Output Layer
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train.reshape(60000, 784), y_train, epochs=3)
    val_loss, val_acc = model.evaluate(x_test.reshape(-1, 784), y_test)
    print(val_acc)
    predictions = model.predict(x_test.reshape(-1, 784))
    print(predictions[20])
    pred_20 = predictions[20]

    max_20 = np.argmax(pred_20)

    print(max_20)
    matplotlib.pyplot.imshow(x_test[20])
    matplotlib.pyplot.show()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
