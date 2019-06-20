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
from keras.models import model_from_json
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras import backend as K

from PIL import Image

from keras.models import load_model
import glob


loaded_model = load_model("first-try.h5")
print("Loaded model from disk")
image_list = []

#iterate all folders:


#build the string and reverse it 
finallicense="" 
filename = "E:\\LeoPrat\\Documents\\License Plate Recognition Git\\LicensePlateRecognition\\Project1\\char-found\\100"
print(filename)
for image in os.listdir(filename):
    print(image)
    img=Image.open(filename + "\\" + image).convert('L')
    img=np.invert(img)
    plt.imshow( img ,cmap='Greys')
    plt.show()
    im2arr = np.array(img)
    im2arr = im2arr.reshape(-1,28,28,1)
    y_pred = loaded_model.predict_classes(im2arr)
    #print(y_pred)
    finallicense=finallicense + str(y_pred)
print("///////////////////////////////////////////////")
print("TARGA FINALE: " + str(finallicense))
print("///////////////////////////////////////////////")

    