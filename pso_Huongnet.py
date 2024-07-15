# coding: utf-8

# Example with a convolutional neural network in keras

import time
import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
import matplotlib.pyplot as plt 

from Hyperactive.hyperactive import RandomSearchOptimizer, ParticleSwarmOptimizer
#import hyperactive

#(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# X_train = X_train.astype('float32')/255.0
# X_test = X_test.astype('float32')/255.0

# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)

featureDir = '/home/tower1/osas/3000samp/'
#dir = '/content/gdrive/MyDrive/PAPER HN/Snoring problem/Data/'

pathSnoreFeat = featureDir + 'Snoring'
pathNonSnoreFeat = featureDir + 'NonSnoring'

listSnoreFile = os.listdir(pathSnoreFeat)
listNonSnoreFile = os.listdir(pathNonSnoreFeat)

#Load Data
train_image = []
label = []

# num_feature = 0
for id in range(0, len(listSnoreFile)):
  # if listSnoreFile[id].endswith('.npy'):
  filePath = os.path.join(pathSnoreFeat, listSnoreFile[id])
  img = np.load(filePath, allow_pickle = True)
  train_image.append(img)
    # num_feature += 1
# label.extend([1] * num_feature)
label.extend([1] * len(listSnoreFile))
X_snore, y_snore = np.array(train_image), np.array(label)

train_img = []
lb = []
# num_feature = 0
for id in range(0, len(listNonSnoreFile)):
  # if listNonSnoreFile[id].endswith('.npy'):
  filePath = os.path.join(pathNonSnoreFeat, listNonSnoreFile[id])
  img = np.load(filePath, allow_pickle = True)
  train_img.append(img)
    # num_feature += 1
# label.extend([0] * num_feature)
lb.extend([0] * len(listNonSnoreFile))
X_non, y_non = np.array(train_img), np.array(lb)

X = np.concatenate((X_snore, X_non), axis = 0)
y = np.concatenate((y_snore, y_non), axis = 0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15)
print(y_train.shape)

sgd = keras.optimizers.SGD(lr=0.01)
adam = keras.optimizers.Adam(lr=0.01)

# this defines the structure of the model and print("time: {}".format(t2-t1))the search space in each layer
search_config = {
    "keras.compile.0": {"loss": ["binary_crossentropy"], "optimizer": [adam, sgd]},
    "keras.fit.0": {"epochs": [10], "batch_size": range(10, 101), "verbose": [2]},
    "keras.layers.Conv2D.1": {
        "filters": range(4, 101),
        "kernel_size": [3, 5, 7],
        "padding": ["same"],
        "activation": ["sigmoid", "relu", "tanh"],
        "input_shape": [(97, 32, 1)],
    },
        "keras.layers.Conv2D.2": {
        "filters": range(4, 101),
        "kernel_size": [3, 5, 7],
        "padding": ["valid"],
        "activation": ["sigmoid", "relu", "tanh"],
    },
    "keras.layers.MaxPooling2D.3": {"pool_size": [(2, 2)]},
    "keras.layers.Dropout.4": {"rate": [0.25]},
    "keras.layers.Conv2D.5": {
        "filters": range(4, 101),
        "kernel_size": [3, 5, 7],
        "padding": ["same"],
        "activation": ["sigmoid", "relu", "tanh"],
    },
        "keras.layers.Conv2D.6": {
        "filters": range(4, 101),
        "kernel_size": [3, 5, 7],
        "padding": ["valid"],
        "activation": ["sigmoid", "relu", "tanh"],
    },
    "keras.layers.MaxPooling2D.7": {"pool_size": [(2, 2)]},
    "keras.layers.Dropout.8": {"rate": [0.25]},
    "keras.layers.Flatten.9": {},
    "keras.layers.Dense.10": {"units": range(4, 101), "activation": ["sigmoid", "relu", "tanh"]},
    #"keras.layers.Dropout.11": {"rate": [0.5]},
    "keras.layers.Dense.11": {"units": range(4, 101), "activation": ["sigmoid", "relu", "tanh"]},
    #"keras.layers.Dropout.7": {"rate": list(np.arange(0.2, 0.8, 0.2))},
    "keras.layers.Dense.12": {"units": [1], "activation": ["sigmoid"]},
}

Optimizer = ParticleSwarmOptimizer(search_config, n_iter=10, n_part=10, metric='accuracy', cv=0.8, w=0.7, c_k=2.0, c_s=2.0)

t1 = time.time()
Optimizer.fit(X_train, y_train)
t2 = time.time()

print("time: {}".format(t2-t1))
print(Optimizer.model_best.summary())

hist = Optimizer.model_best.fit(X_train, y_train, validation_data=(X_test, y_test),
                    epochs=10, batch_size=10, verbose=1)

plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.show()
plt.savefig("accuracy.png", dpi = 600)

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.show()
plt.savefig("loss.png", dpi = 600)


# predict from test data
Optimizer.predict(X_test)
score = Optimizer.score(X_test, y_test)

print("test score: {}".format(score))
