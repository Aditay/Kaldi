import numpy
import pickle
action = pickle.load(open("action.pkl", "rb"))
# print action
action = action["action_thriller"]
action = numpy.asarray(action)
print action.shape

animation = pickle.load(open("animation.pkl", "rb"))
print animation
animation = animation["animation"]
animation = numpy.asarray(animation)
print animation.shape

comedy = pickle.load(open("comedy.pkl", "rb"))
comedy = comedy["comedy"]
comedy = numpy.asarray(comedy)
print comedy.shape

horror = pickle.load(open("horror.pkl", "rb"))
horror = horror["horror"]
horror = numpy.asarray(horror)
print horror.shape

romance = pickle.load(open("romance.pkl", "rb"))
romance = romance["romance"]
romance = numpy.asarray(romance)
print romance.shape

scifi = pickle.load(open("scifi.pkl", "rb"))
scifi = scifi["scifi"]
scifi = numpy.asarray(scifi)
print scifi.shape

action_labels = numpy.zeros(72000)
animation_labes = numpy.ones(72000)
print action_labels
print action_labels.shape

comedy_labels = numpy.ones(72000)
comedy_labels[:] = 2
# print comedy_labels
# print comedy_labels.shape
horror_labels = numpy.ones(72000)
horror_labels[:] = 3
romance_labels = numpy.ones(72000)
romance_labels[:] = 4
scifi_labels = numpy.ones(72000)
scifi_labels[:] = 5

comedy_labels = comedy_labels.astype(numpy.int64)
# print comedy_labels
action_labels = action_labels.astype(numpy.int64)
animation_labes = animation_labes.astype(numpy.int64)
horror_labels = horror_labels.astype(numpy.int64)
romance_labels = romance_labels.astype(numpy.int64)
scifi_labels = scifi_labels.astype(numpy.int64)

action_test = action[0:14400]
action_train = action[14400:]
action_labels_test = action_labels[0:14400]
action_labels_train = action_labels[14400:]
print action_labels_test.shape
print action_labels_train.shape

animation_test = animation[0:14400]
animation_train = animation[14400:]
animation_labes_test = animation_labes[0:14400]
animation_labes_train = animation_labes[14400:]
comedy_test = comedy[0:14400]
comedy_train = comedy[14400:]
comedy_labels_test = comedy_labels[0:14400]
comedy_labels_train = comedy_labels[14400:]
horror_test = horror[0:14400]
horror_train = horror[14400:]
horror_labels_test = horror_labels[0:14400]
horror_labels_train = horror_labels[14400:]
romance_test = romance[0:14400]
romance_train = romance[14400:]
romance_labels_test = romance_labels[0:14400]
romance_labels_train = romance_labels[14400:]
scifi_test = scifi[0:14400]
scifi_train = scifi[14400:]
scifi_labels_test = scifi_labels[0:14400]
scifi_labels_train = scifi_labels[14400:]

train_x = numpy.concatenate(
    (action_train, animation_train, comedy_train, horror_train, romance_train, scifi_train), axis=0)
print train_x.shape

test_x = numpy.concatenate(
    (action_test, animation_test, comedy_test, horror_test, romance_test, scifi_test), axis=0
)
print test_x.shape

train_y = numpy.concatenate(
    (action_labels_train, animation_labes_train, comedy_labels_train, horror_labels_train, romance_labels_train, scifi_labels_train), axis=0
)

print train_y.shape

test_y = numpy.concatenate(
    (action_labels_test, animation_labes_test, comedy_labels_test, horror_labels_test, romance_labels_test, scifi_labels_test), axis=0
)

print test_y.shape

numpy.random.seed(0)

# numpy.random.seed(0)
# a = numpy.arange(9)
# b = numpy.arange(9)
# c = numpy.arange(9)
# print a
# print b
# a = numpy.random.permutation(a)
# b = numpy.random.permutation(b)
# print a
# print b
# print c[a]
# c = c[a]
# print c

numpy.random.seed(0)
a = numpy.arange(345600)
# print a
a = numpy.random.permutation(a)
# print a
train_x = train_x[a]

train_y = train_y[a]
print train_y.shape
print train_x.shape

from keras.models import Sequential
from keras import models
from keras.layers import Dense, Dropout, Activation
from keras.utils import np_utils
train_y = np_utils.to_categorical(train_y, 6)
print train_y.shape

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D

# train_x = train_x / 255.0
# test_x = test_x / 255.0

model = Sequential()
model.add(Convolution2D(96,11,3, border_mode='same',input_shape=(100,150,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Convolution2D(256,5,1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Convolution2D(384,3,1))
# model.add(Convolution2D(384,3,1))
# model.add(Convolution2D(256,3,1))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dense(1024, activation='relu'))
model.add(Dense(6, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adagrad', metrics=['accuracy'])

model.fit(train_x, train_y, nb_epoch=50, batch_size=1000)
model.save('genre_classifier.h5')

score = model.evaluate(test_x, test_y, verbose=0)
print score

