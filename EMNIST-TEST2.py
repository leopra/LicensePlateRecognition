from __future__ import absolute_import, division, print_function


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

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder



# read data (digits and letters) --> all together
img_shape = 28 # size of the MNIST images
temp_train_images, temp_train_labels = loadlocal_mnist(
        images_path='emnist-byclass-train-images-idx3-ubyte', 
        labels_path='emnist-byclass-train-labels-idx1-ubyte')

temp_test_images, temp_test_labels = loadlocal_mnist(
        images_path='emnist-byclass-test-images-idx3-ubyte', 
        labels_path='emnist-byclass-test-labels-idx1-ubyte')

ims = []
labs = []

train_index = []
counter = 0
print('Dimensions: %s' % (temp_train_images.shape[0]))
for i in range(temp_train_images.shape[0]):
    if ((temp_train_labels[i] < 36) and (temp_train_labels[i] != 24) and (temp_train_labels[i] != 18) and (temp_train_labels[i] != 26)):
        counter+=1
        if (temp_train_labels[i] == 19):
            temp_train_labels[i] = 18
        elif (temp_train_labels[i] == 20):
            temp_train_labels[i] = 19
        elif (temp_train_labels[i] == 21):
            temp_train_labels[i] = 20
        elif (temp_train_labels[i] == 22):
            temp_train_labels[i] = 21
        elif (temp_train_labels[i] == 23):
            temp_train_labels[i] = 22
        elif (temp_train_labels[i] == 25):
            temp_train_labels[i] = 23
        elif (temp_train_labels[i] == 27):
            temp_train_labels[i] = 24
        elif (temp_train_labels[i] == 28):
            temp_train_labels[i] = 25
        elif (temp_train_labels[i] == 29):
            temp_train_labels[i] = 26
        elif (temp_train_labels[i] == 30):
            temp_train_labels[i] = 27
        elif (temp_train_labels[i] == 31):
            temp_train_labels[i] = 28
        elif (temp_train_labels[i] == 32):
            temp_train_labels[i] = 29
        elif (temp_train_labels[i] == 33):
            temp_train_labels[i] = 30
        elif (temp_train_labels[i] == 34):
            temp_train_labels[i] = 31
        elif (temp_train_labels[i] == 35):
            temp_train_labels[i] = 32
        elif (temp_train_labels[i] == 36):
            temp_train_labels[i] = 33
        ims.append(temp_train_images[i])
        labs.append(temp_train_labels[i])

train_images = np.array(ims).reshape((-1, img_shape, img_shape, 1), order="F")
train_labels = np.array(labs).reshape((-1))
print(counter)

ims = []
labs = []

counter = 0
print('Dimensions: %s' % (temp_test_images.shape[0]))
for i in range(temp_test_images.shape[0]):
    if ((temp_test_labels[i] < 36) and (temp_test_labels[i] != 24) and (temp_test_labels[i] != 18) and (temp_test_labels[i] != 26)):
        counter+=1
        if (temp_test_labels[i] == 19):
            temp_test_labels[i] = 18
        elif (temp_test_labels[i] == 20):
            temp_test_labels[i] = 19
        elif (temp_test_labels[i] == 21):
            temp_test_labels[i] = 20
        elif (temp_test_labels[i] == 22):
            temp_test_labels[i] = 21
        elif (temp_test_labels[i] == 23):
            temp_test_labels[i] = 22
        elif (temp_test_labels[i] == 25):
            temp_test_labels[i] = 23
        elif (temp_test_labels[i] == 27):
            temp_test_labels[i] = 24
        elif (temp_test_labels[i] == 28):
            temp_test_labels[i] = 25
        elif (temp_test_labels[i] == 29):
            temp_test_labels[i] = 26
        elif (temp_test_labels[i] == 30):
            temp_test_labels[i] = 27
        elif (temp_test_labels[i] == 31):
            temp_test_labels[i] = 28
        elif (temp_test_labels[i] == 32):
            temp_test_labels[i] = 29
        elif (temp_test_labels[i] == 33):
            temp_test_labels[i] = 30
        elif (temp_test_labels[i] == 34):
            temp_test_labels[i] = 31
        elif (temp_test_labels[i] == 35):
            temp_test_labels[i] = 32
        elif (temp_test_labels[i] == 36):
            temp_test_labels[i] = 33
        
        temp = temp_test_images[i]
#             np.reshape(temp, (28,28))
        ims.append(temp)
        labs.append(temp_test_labels[i])

test_images = np.array(ims).reshape((-1, img_shape, img_shape, 1), order="F")
test_labels = np.array(labs).reshape((-1))
print(counter)

#     np.reshape(train_images, (train_labels.shape[0], 28*28))
print(train_images.shape)
print('Dimensions: %s' % (train_labels.shape))
#     print('\n1st row', train_images[0])
print(test_images.shape)
print('Dimensions: %s' % (test_labels.shape))


#     print('Digits:  0 1 2 3 4 5 6 7 8 9')
print('labels: %s' % np.unique(train_labels))
#     print('Class distribution: %s' % np.bincount(train_labels))

# permute data
ord = np.random.permutation(test_labels.shape[0])
test_images = test_images[ord]
test_labels = test_labels[ord]
x_train, y_train, x_test, y_test = train_images, train_labels, test_images, test_labels
batch_size = 128
epochs = 6
#     epochs = 8

# train set
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(y_train)
#     print(integer_encoded)
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded_train = onehot_encoder.fit_transform(integer_encoded)
#     print(onehot_encoded_train)

# test set
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(y_test)
#     print(integer_encoded)

onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded_test = onehot_encoder.fit_transform(integer_encoded)
#     print(onehot_encoded_test)

print("Size of:")
print("- Training-set:\t\t{}".format(x_train.shape[0]))
print("- Test-set:\t\t{}".format(x_test.shape[0]))

#     num_classes = 36
num_classes = 33
class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J',
                'K', 'L', 'M', 'N', 'P', 'R', 'S', 'T',
                'U', 'V', 'W', 'X', 'Y', 'Z']#, 'a', 'b', 'c', 'd', 
#                    'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 
#                    'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x',
#                    'y', 'z' ]

# Data preprocessing: image normalization
x_train = x_train / 255.0

x_test = x_test / 255.0

# Plot images
#plot_images(x_test[:9], y_test[:9], class_names)

# Create Keras model and evaluate its performance
img_rows, img_cols = 28, 28

if K.image_data_format() == "channel_first":
    x_train = x_train.reshape(x_train_shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
    activation='relu',
    input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=num_classes, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])

model.summary()

history = model.fit(x_train, onehot_encoded_train,
    epochs=epochs,
    verbose=1,
    batch_size=batch_size,
    validation_data=(x_test, onehot_encoded_test))
score = model.evaluate(x_test, onehot_encoded_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

predictions = model.predict(x_test, verbose=1)

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model6_cut.h5")
print("Saved model to disk")

# fashion utilities
#plot_example_errors(model, x_test, y_test, class_names)
#image_with_PDF(x_test, y_test, predictions, num_classes, class_names)

#lot_history([('CNN_classifier', history)])
