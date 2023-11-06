import matplotlib.pyplot as plt
import numpy as np
from keras.models import *
from keras.layers import *
from keras.datasets import mnist
import os

OUT_DIR = './DNN_out/'
img_shape = (28, 28, 1)

epochs = 100000
batch_size = 128
noise = 100
sample_interval = 100

(X_train, _), (_, _) = mnist.load_data()
print(X_train.shape)

X_train = X_train / 127.5 - 1
X_train = np.expand_dims(X_train, axis = 3)
print(X_train.shape)

generator = Sequential()
generator.add(Dense(128, input_dim=noise))
# relu는 -값을 0으로, 양수값은 그대로. 그래서 여기선 안씀 (데이터 손실)
# LeakyReLU는 음수값을 어느 정도 가져옴 
generator.add(LeakyReLU(alpha=0.1))
generator.add(Dense(784, activation='tanh'))
