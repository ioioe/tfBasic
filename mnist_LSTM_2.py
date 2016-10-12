'''
My model based on Google's: scan row first and then column
'''

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation, Permute
from keras.layers import LSTM
from keras.initializations import normal, identity
from keras.optimizers import RMSprop, Adam
from keras.utils import np_utils
from setGPU import set_gpu_memory_keras

set_gpu_memory_keras(0.4)

batch_size = 128
nb_classes = 10
nb_epochs = 200
hidden_units = 128
n_input = 28
n_steps = 28

learning_rate = 1e-3

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], n_steps, n_input)
X_test = X_test.reshape(X_test.shape[0], n_steps, n_input)
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

model.add(LSTM(output_dim=28,
               activation='tanh',
               return_sequences=True,
               input_shape=(n_steps, n_input)))

model.add(Permute((2, 1)))

model.add(LSTM(output_dim=hidden_units,
               activation='tanh'))

model.add(Dense(nb_classes))
model.add(Activation('softmax'))
adam = Adam(lr=learning_rate)
model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epochs,
          verbose=1, validation_data=(X_test, Y_test))

scores = model.evaluate(X_test, Y_test, verbose=0)
print('IRNN test score:', scores[0])
print('IRNN test accuracy:', scores[1])