
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




#load the dictionary from the summary txt
dmapping = {}
with open("E:\\LeoPrat\\Documents\\License Plate Recognition Git\\LicensePlateRecognition\\train-class\\emnist-byclass-mapping.txt") as f:
    for line in f:
        (key, val) = line.split()
        dmapping[int(key)] = val

print (dmapping)

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model6_cut.h5") # load weights

image_list = []

#create a dictionary to print the right value
ascii_dict = dict()
ascii_in_number = range(0,256)
for i in ascii_in_number:
    ascii_dict[str(i)] = chr(i)
print(ascii_dict)
#iterate all folders:

folder=0
#for folder in range(0,81) :
filename = "E:\\LeoPrat\\Documents\\License Plate Recognition Git\\LicensePlateRecognition\\Project1\\char-found\\100"  #+ str(folder)
print(filename)
finallicense="" 

#build the string and reverse it 
for image in os.listdir(filename):
    print(image)
    img=Image.open(filename + "\\" + image).convert('L')
    #img= img.rotate(90)
    plt.imshow(img, cmap='Greys')
    plt.show
    img=np.invert(img)
    plt.imshow( img ,cmap='Greys')
    plt.show()
    im2arr = np.array(img)
    im2arr = im2arr.reshape(-1,28,28,1)
    y_pred = loaded_model.predict_classes(im2arr)
    asciiword=""
    for name, value in dmapping.items():  
        if name == y_pred:
            asciiword= value
            print("value: " + str(value))
            for namek, valuek in ascii_dict.items():  
                if namek == asciiword:
                    print(valuek)
                    #add carachter to final plate
                    finallicense=finallicense + str(valuek)


print("///////////////////////////////////////////////")
finallicense = finallicense[::-1] 
print("TARGA FINALE: " + str(finallicense))
print("///////////////////////////////////////////////")                        