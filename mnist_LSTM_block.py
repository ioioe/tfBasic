'''
My re-implementation of block-based LSTM model
'''

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation, Bidirectional
from keras.layers import LSTM
from keras.initializations import normal, identity
from keras.optimizers import RMSprop, Adam
from keras.utils import np_utils
from setGPU import set_gpu_memory_keras
import numpy as np


def block_reshape(x, block_h, block_w):
    n = x.shape[0]
    h = x.shape[1]
    w = x.shape[2]
    x_reshaped = np.array([])
    for i in range(n):
        print i, x_reshaped.shape
        sub_x = np.empty((0, block_h*block_w))
        for j in range(h / block_h):
            if j % 2 == 0:
                # left to right
                for k in range(w / block_w):
                    sub_x = np.concatenate((sub_x,
                                            x[i, j*block_h: (j+1)*block_h, k*block_w: (k+1)*block_w].reshape(1, block_h*block_w)),
                                           axis=0)
            else:
                # right to left
                for k in reversed(range(w / block_w)):
                    sub_x = np.concatenate((sub_x,
                                            x[i, j*block_h: (j+1)*block_h, k*block_w: (k+1)*block_w].reshape(1, block_h*block_w)),
                                           axis=0)
        sub_x = sub_x.reshape(1, -1, block_h*block_w)
        if i == 0:
            x_reshaped = sub_x
        else:
            x_reshaped = np.concatenate((x_reshaped, sub_x), axis=0)
    return x_reshaped


set_gpu_memory_keras(0.4)

batch_size = 128
nb_classes = 10
nb_epochs = 200
hidden_units = 128
n_block_h = 4
n_block_w = 7
n_input = n_block_h * n_block_w
n_steps = 28

learning_rate = 1e-3

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], n_steps, n_input)
X_test = X_test.reshape(X_test.shape[0], n_steps, n_input)
print X_train.shape
X_train = block_reshape(X_train, n_block_h, n_block_w)
X_test = block_reshape(X_test, n_block_h, n_block_w)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

print('Evaluate row-based LSTM...')
model = Sequential()

model.add(LSTM(output_dim=hidden_units, activation='tanh', input_shape=(n_steps, n_input)))

model.add(Dense(nb_classes))
model.add(Activation('softmax'))
# rmsprop = RMSprop(lr=learning_rate)
adam = Adam(lr=learning_rate)
model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epochs,
          verbose=1, validation_data=(X_test, Y_test))

scores = model.evaluate(X_test, Y_test, verbose=0)
print('IRNN test score:', scores[0])
print('IRNN test accuracy:', scores[1])