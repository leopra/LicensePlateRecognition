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



# read data (digits and letters) --> all together
img_shape = 28 # size of the MNIST images
temp_train_images, temp_train_labels = loadlocal_mnist(
        images_path='train-class/emnist-byclass-train-images-idx3-ubyte', 
        labels_path='train-class/emnist-byclass-train-labels-idx1-ubyte')

temp_test_images, temp_test_labels = loadlocal_mnist(
        images_path='train-class/emnist-byclass-test-images-idx3-ubyte', 
        labels_path='train-class/emnist-byclass-test-labels-idx1-ubyte')
i=0
for image in temp_test_images:
    image_index=i
    if (temp_train_labels[image_index]==22) :
        plt.imshow(temp_train_images[image_index].reshape(28,28),cmap='Greys')
        plt.show()
        print(temp_train_labels[image_index])
    i+=1
    
#print(labs[image_index])
# ims = []
# labs = []

# train_index = []
# counter = 0
# print('Dimensions: %s' % (temp_train_images.shape[0]))
# for i in range(temp_train_images.shape[0]):
#     if ((temp_train_labels[i] < 36) and (temp_train_labels[i] != 24) and (temp_train_labels[i] != 18) and (temp_train_labels[i] != 26)):
#         counter+=1
#         if (temp_train_labels[i] == 19):
#             temp_train_labels[i] = 18
#         elif (temp_train_labels[i] == 20):
#             temp_train_labels[i] = 19
#         elif (temp_train_labels[i] == 21):
#             temp_train_labels[i] = 20
#         elif (temp_train_labels[i] == 22):
#             temp_train_labels[i] = 21
#         elif (temp_train_labels[i] == 23):
#             temp_train_labels[i] = 22
#         elif (temp_train_labels[i] == 25):
#             temp_train_labels[i] = 23
#         elif (temp_train_labels[i] == 27):
#             temp_train_labels[i] = 24
#         elif (temp_train_labels[i] == 28):
#             temp_train_labels[i] = 25
#         elif (temp_train_labels[i] == 29):
#             temp_train_labels[i] = 26
#         elif (temp_train_labels[i] == 30):
#             temp_train_labels[i] = 27
#         elif (temp_train_labels[i] == 31):
#             temp_train_labels[i] = 28
#         elif (temp_train_labels[i] == 32):
#             temp_train_labels[i] = 29
#         elif (temp_train_labels[i] == 33):
#             temp_train_labels[i] = 30
#         elif (temp_train_labels[i] == 34):
#             temp_train_labels[i] = 31
#         elif (temp_train_labels[i] == 35):
#             temp_train_labels[i] = 32
#         elif (temp_train_labels[i] == 36):
#             temp_train_labels[i] = 33
#         ims.append(temp_train_images[i])
#         labs.append(temp_train_labels[i])

# train_images = np.array(ims).reshape((-1, img_shape, img_shape, 1), order="F")
# train_labels = np.array(labs).reshape((-1))
# print(counter)

# ims = []
# labs = []

# counter = 0
# print('Dimensions: %s' % (temp_test_images.shape[0]))
# for i in range(temp_test_images.shape[0]):
#     if ((temp_test_labels[i] < 36) and (temp_test_labels[i] != 24) and (temp_test_labels[i] != 18) and (temp_test_labels[i] != 26)):
#         counter+=1
#         if (temp_test_labels[i] == 19):
#             temp_test_labels[i] = 18
#         elif (temp_test_labels[i] == 20):
#             temp_test_labels[i] = 19
#         elif (temp_test_labels[i] == 21):
#             temp_test_labels[i] = 20
#         elif (temp_test_labels[i] == 22):
#             temp_test_labels[i] = 21
#         elif (temp_test_labels[i] == 23):
#             temp_test_labels[i] = 22
#         elif (temp_test_labels[i] == 25):
#             temp_test_labels[i] = 23
#         elif (temp_test_labels[i] == 27):
#             temp_test_labels[i] = 24
#         elif (temp_test_labels[i] == 28):
#             temp_test_labels[i] = 25
#         elif (temp_test_labels[i] == 29):
#             temp_test_labels[i] = 26
#         elif (temp_test_labels[i] == 30):
#             temp_test_labels[i] = 27
#         elif (temp_test_labels[i] == 31):
#             temp_test_labels[i] = 28
#         elif (temp_test_labels[i] == 32):
#             temp_test_labels[i] = 29
#         elif (temp_test_labels[i] == 33):
#             temp_test_labels[i] = 30
#         elif (temp_test_labels[i] == 34):
#             temp_test_labels[i] = 31
#         elif (temp_test_labels[i] == 35):
#             temp_test_labels[i] = 32
#         elif (temp_test_labels[i] == 36):
#             temp_test_labels[i] = 33
        
#         temp = temp_test_images[i]
# #             np.reshape(temp, (28,28))
#         ims.append(temp)
#         labs.append(temp_test_labels[i])

# test_images = np.array(ims).reshape((-1, img_shape, img_shape, 1), order="F")
# test_labels = np.array(labs).reshape((-1))
# print(counter)



# #     np.reshape(train_images, (train_labels.shape[0], 28*28))
# print(train_images.shape)
# print('Dimensions: %s' % (train_labels.shape))
# #     print('\n1st row', train_images[0])
# print(test_images.shape)
# print('Dimensions: %s' % (test_labels.shape))
# # load json and create model
# json_file = open('model.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)
# # load weights into new model
# loaded_model.load_weights("model6_cut.h5")
# print("Loaded model from disk")
# image_list = []

# for filename in glob.glob('E:\\LeoPrat\\Documents\\License Plate Recognition Git\\LicensePlateRecognition\\char-found\\*.jpg'):
#     #convert to gray-scale as EMNIST images
#     img=Image.open(filename) .convert('LA')
#     print("carica: ")
#     print(filename)
#     image_list.append(img)
#     im2arr = np.array(img)
#     im2arr = im2arr.reshape(-1,28,28,1)
#     y_pred = loaded_model.predict_classes(im2arr)

#     print(y_pred)