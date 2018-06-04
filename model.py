import time

def taketime(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    print("%02d:%02d:%02d" % (h, m, s))

starttime = time.time()

import csv
import cv2
import numpy as np


lines = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

cftime = time.time()
taketime(cftime - starttime)
print('...Csv File Loaded')

images = []
measurements = []

# Three Camara
correction = 0.2
for line in lines:
    steering_center = float(line[3])
    steering_left = steering_center + correction
    steering_right = steering_center - correction
    measurements.extend([steering_center, steering_left, steering_right])

    for i in range(3):
        source_path = line[i]
        filename = source_path.split('/')[-1]
        current_path = './data/IMG/' + filename
        image = cv2.imread(current_path)
        images.append(image)

tctime = time.time()
taketime(tctime - starttime)
print('...Three Camara Images Loaded')

# flip
augmented_images, augmented_measurements = [], []
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image, 1))
    augmented_measurements.append(measurement * (-1.0))

fdtime = time.time()
taketime(fdtime - starttime)
print('...Flipping Done')

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D

print('...Keras go go go')

#  Flatten Network
# model = Sequential()
# model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape = (160,320,3)))
# model.add(Flatten())
# model.add(Dense(1))

# LeNet-5
# model = Sequential()
# model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape = (160,320,3)))
# model.add(Cropping2D(cropping = ((75, 25), (0, 0))))
# model.add(Convolution2D(6,5,5, activation = 'relu'))
# model.add(MaxPooling2D())
# model.add(Convolution2D(6,5,5, activation = 'relu'))
# model.add(MaxPooling2D())
# model.add(Flatten())
# model.add(Dense(120))
# model.add(Dense(84))
# model.add(Dense(1))

# Nvidia Network
model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape = (160,320,3)))
model.add(Cropping2D(cropping = ((70, 20), (0, 0))))
model.add(Convolution2D(24,5,5, subsample=(2,2), activation = 'relu'))
model.add(Convolution2D(36,5,5, subsample=(2,2), activation = 'relu'))
model.add(Convolution2D(48,5,5, subsample=(2,2), activation = 'relu'))
model.add(Convolution2D(64,3,3, activation = 'relu'))
model.add(Convolution2D(64,3,3, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss = 'mse', optimizer = 'adam')
model.fit(X_train, y_train, validation_split = 0.2, shuffle = True, nb_epoch = 3, verbose = 1)

model.save('model_3C.h5')

alltime = time.time()
taketime(alltime - starttime)
print('...Model Saved')
print('Training is Done!')

exit()