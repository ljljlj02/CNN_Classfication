import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
import os
from keras.preprocessing import image
from keras.utils import np_utils
from keras.models import load_model
cwd = os.getcwd()
datafile = cwd + '/CIFAR_10_normal.pkl' #标准化后
(X_train, y_train), (X_test, y_test) = pickle.load(open(datafile,"rb"))
y_train = np_utils.to_categorical(y_train,10)
y_test = np_utils.to_categorical(y_test,10)
model = load_model('./CIFAR10_normal/CIFAR10_model.h5')
result = model.evaluate(x=X_train,y=y_train,batch_size=128)
print(result[0])
print(result[1])
result = model.evaluate(x=X_test,y=y_test,batch_size=128)
print(result[0])
print(result[1])
