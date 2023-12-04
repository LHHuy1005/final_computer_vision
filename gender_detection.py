from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Activation, Flatten, Dropout, Dense
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input, Reshape
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import random
import cv2
import os
import glob

lr = 1e-3
img_dims = (96,96,3)

img_width = 96
img_height = 96

x_train = []
y_train = []

for path in glob.glob('gender_dataset_face/Training/female/*'):
    img = cv2.imread(path)
    img_new = cv2.resize(img, (img_width, img_height))
    x_train.append(img_new)
    y_train.append(1)
    
for path in glob.glob('gender_dataset_face/Training/male/*'):
    img = cv2.imread(path)
    img_new = cv2.resize(img, (img_width, img_height))
    x_train.append(img_new)
    y_train.append(0)

# normalize data
x_train = np.array(x_train, dtype="float") / 255.0
y_train = np.array(y_train)

x_tst = []
y_tst = []

for path in glob.glob('gender_dataset_face\Validation\\female\*'):
    img = cv2.imread(path)
    img = cv2.resize(img, (img_width, img_height))
    img = np.array(img) / 255.0
    x_tst.append(img)
    y_tst.append(0)

for path in glob.glob('gender_dataset_face\Validation\male\*'):
    img = cv2.imread(path)
    img = cv2.resize(img, (img_width, img_height))
    img = np.array(img) / 255.0
    x_tst.append(img)
    y_tst.append(1)

x_test = np.array(x_tst)
y_test = np.array(y_tst)

# split dataset for training and validation
# (trainX, testX, trainY, testY) = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
#
# trainY = to_categorical(trainY, num_classes=2) # [[1, 0], [0, 1], [0, 1], ...]
# testY = to_categorical(testY, num_classes=2)
#
# # augmenting datset
# aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
#                          height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
#                          horizontal_flip=True, fill_mode="nearest")
#
# # define model
# def build(width, height, depth, classes):
#     model = Sequential()
#     inputShape = (height, width, depth)
#     chanDim = -1
#
#     if K.image_data_format() == "channels_first": #Returns a string, either 'channels_first' or 'channels_last'
#         inputShape = (depth, height, width)
#         chanDim = 1
#
#     # The axis that should be normalized, after a Conv2D layer with data_format="channels_first",
#     # set axis=1 in BatchNormalization.
#
#     model.add(Conv2D(32, (3,3), padding="same", input_shape=inputShape))
#     model.add(Activation("relu"))
#     model.add(BatchNormalization(axis=chanDim))
#     model.add(MaxPooling2D(pool_size=(3,3)))
#     model.add(Dropout(0.25))
#
#     model.add(Conv2D(64, (3,3), padding="same"))
#     model.add(Activation("relu"))
#     model.add(BatchNormalization(axis=chanDim))
#
#     model.add(Conv2D(64, (3,3), padding="same"))
#     model.add(Activation("relu"))
#     model.add(BatchNormalization(axis=chanDim))
#     model.add(MaxPooling2D(pool_size=(2,2)))
#     model.add(Dropout(0.25))
#
#     model.add(Conv2D(128, (3,3), padding="same"))
#     model.add(Activation("relu"))
#     model.add(BatchNormalization(axis=chanDim))
#
#     model.add(Conv2D(128, (3,3), padding="same"))
#     model.add(Activation("relu"))
#     model.add(BatchNormalization(axis=chanDim))
#     model.add(MaxPooling2D(pool_size=(2,2)))
#     model.add(Dropout(0.25))
#
#     model.add(Flatten())
#     model.add(Dense(1024))
#     model.add(Activation("relu"))
#     model.add(BatchNormalization())
#     model.add(Dropout(0.5))
#
#     model.add(Dense(classes))
#     model.add(Activation("sigmoid"))
#
#     return model
#
# # build model
# model = build(width=img_dims[0], height=img_dims[1], depth=img_dims[2],
#                             classes=2)



# CNN model
inp = Input(shape=(img_height, img_width, 3))  # input shape
cnn = Conv2D(filters=8, kernel_size=3, activation='relu')(inp)
pooling = MaxPooling2D(pool_size=(2, 2))(cnn)
drop = Dropout(0.2)(pooling)

cnn = Conv2D(filters=32, kernel_size=4, activation='relu')(drop)
pooling = MaxPooling2D(pool_size=(2, 2))(cnn)
drop = Dropout(0.2)(pooling)

cnn = Conv2D(filters=16, kernel_size=4, activation='relu')(drop)
pooling = MaxPooling2D(pool_size=(2, 2))(cnn)

f = Flatten()(pooling)
fc1 = Dense(units=32, activation='relu')(f)
fc2 = Dense(units=16, activation='relu')(fc1)
out = Dense(units=1, activation='sigmoid')(fc2)

model = Model(inputs=inp, outputs=out)
model.summary()

# compile the model
optimizer = tf.keras.optimizers.Adam(lr=lr, decay=lr/64)
model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])

# train the model
history = model.fit_generator(aug.flow(trainX, trainY, batch_size=64),
                        validation_data=(testX,testY),
                        steps_per_epoch=len(trainX) // 64,
                        epochs=100, verbose=1)

# save the model to disk
model.save('gender_detection.model')

# Plot accuracy
his = history
plt.plot(his.history['accuracy'])
plt.plot(his.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# "Loss"
plt.plot(his.history['loss'])
plt.plot(his.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()