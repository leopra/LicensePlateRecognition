#first mnist try 
import numpy as np
np.random.seed(123)  # for reproducibility
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from PIL import Image

from keras.models import load_model
import glob
image_list = []
model = load_model('first-try.h5')

for filename in glob.glob('E:\\LeoPrat\\Documents\\License Plate Recognition Git\\LicensePlateRecognition\\char-found\\*.jpg'):
    img=Image.open(filename).convert('L')
    print("carica: ")
    print(img)
    image_list.append(img)
    im2arr = np.array(img)
    im2arr = im2arr.reshape(1,28,28,1)
    y_pred = model.predict_classes(im2arr)
    print(y_pred)
#img = Image.open('E:\\LeoPrat\\Documents\\License Plate Recognition Git\\LicensePlateRecognition\\char-found\\288char.jpg').convert("L")
#img = np.resize(img, (28,28,1))