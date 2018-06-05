import time

def taketime(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return "%02d:%02d:%02d" % (h, m, s)

starttime = time.time() # model start time

import csv
import cv2
import numpy as np

# Load The Records From 'driving_log.csv'
lines = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

cftime = time.time()
print(taketime(cftime - starttime) + '...\'driving_log.csv\' Loaded')

images = []
measurements = []

# Three Camara (Center, left, right)
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
print(taketime(tctime - starttime) + '...Three Camara Images Loaded')

# Flip All Images Horizontal Symmetry
augmented_images, augmented_measurements = [], []
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image, 1))
    augmented_measurements.append(measurement * (-1.0))

fdtime = time.time()
print(taketime(fdtime - starttime) + '...Flipping Done')

X_train = np.array(augmented_images) # images
y_train = np.array(augmented_measurements) # angles

# from sklearn.utils import shuffle
# X_train, y_train = shuffle(X_train, y_train) # shuffle training data

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
# from keras.utils import plot_model # network visualization

print('...Keras go go go')

## Flatten Network ##
# model = Sequential()
# model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape = (160,320,3)))
# model.add(Flatten())
# model.add(Dense(1))

## LeNet-5 ##
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

## Nvidia Network ##
model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape = (160,320,3))) # nomalize and 0 mean
model.add(Cropping2D(cropping = ((70, 20), (0, 0)))) # crop useless pixels of image
model.add(Convolution2D(24,5,5, subsample=(2,2), activation = 'relu'))
model.add(Convolution2D(36,5,5, subsample=(2,2), activation = 'relu'))
model.add(Convolution2D(48,5,5, subsample=(2,2), activation = 'relu'))
model.add(Convolution2D(64,3,3, activation = 'relu'))
model.add(Convolution2D(64,3,3, activation = 'relu'))
model.add(Dropout(0.2)) # add a dropout to reduce overfitting
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss = 'mse', optimizer = 'adam')
model.fit(X_train, y_train, batch_size = 256, validation_split = 0.2, shuffle = True, nb_epoch = 7, verbose = 1)

model.save('model.h5')

alltime = time.time()
print(taketime(alltime - starttime) + '...Model Saved') 
print('Training is Done!')

# plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)
# print('Visualization of the architecture is saved.')

exit()