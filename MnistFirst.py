#first mnist try 
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import scipy
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import os

from mlxtend.data import loadlocal_mnist
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras import backend as K

from PIL import Image

from keras.models import load_model
import glob
import numpy as np
np.random.seed(123)  # for reproducibility
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils

from keras.datasets import mnist
 
# Load pre-shuffled MNIST data into train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

print(X_train.shape)
# (60000, 28, 28)


#fix error
#https://stackoverflow.com/questions/41651628/negative-dimension-size-caused-by-subtracting-3-from-1-for-conv2d
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

print("shape: " + str(X_train.shape[0]))
# (60000, 1, 28, 28)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

print(y_train.shape)
# (60000,)
i=0
print(y_train[:10])
# [5 0 4 1 9 2 1 3 1 4]
# for image in X_train:
#     image_index=i
#     if (y_train[image_index]==8) :
#         plt.imshow(X_train[image_index].reshape(28,28) ,cmap='Greys')
#         plt.show()
#         print(y_train[image_index])
#     i+=1

# Convert 1-dimensional class arrays to 10-dimensional class matrices
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

model = Sequential()
model.add(Convolution2D(32, 3, 3, activation='relu', input_shape=(28,28,1)))
print(model.output_shape)
model.add(Convolution2D(32, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X_train, Y_train, 
          batch_size=32, nb_epoch=10, verbose=1)

score = model.evaluate(X_test, Y_test, verbose=0)

model.save('first-try.h5')
