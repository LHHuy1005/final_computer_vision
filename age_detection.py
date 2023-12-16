import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from zipfile import ZipFile
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.layers import Conv2D, AveragePooling2D, GlobalAveragePooling2D
from tensorflow.keras import utils
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint

# Setting random seeds to reduce the amount of randomness in the neural net weights and results
# The results may still not be exactly reproducible

np.random.seed(42)
tf.random.set_seed(42)

img_width = 200
img_height = 200

# # Unzipping the dataset file combined_faces.zip

# combined_faces_zip_path = "age_dataset_face\combined_faces.zip"

# with ZipFile(combined_faces_zip_path, 'r') as myzip:
#     myzip.extractall()
#     print('Done unzipping combined_faces.zip')
    
# # Unzipping the dataset file combined_faces_train_augmented.zip

# combined_faces_zip_path = "age_dataset_face\combined_faces_train_augmented.zip"

# with ZipFile(combined_faces_zip_path, 'r') as myzip:
#     myzip.extractall()
#     print('Done unzipping combined_faces_train_augmented.zip')

# Importing the augmented training dataset and testing dataset to create tensors of images using the filename paths.

train_aug_df = pd.read_csv("age_dataset_face\images_filenames_labels_train_augmented.csv")
test_df = pd.read_csv("age_dataset_face\images_filenames_labels_test.csv")


# Defining a function to return the class labels corresponding to the re-distributed 7 age-ranges.
def class_labels_reassign(age):

    if 1 <= age <= 9:
        return 0
    elif 10 <= age <=27:
        return 1
    elif 28 <= age <= 65:
        return 2
    else:
        return 3
    
    
train_aug_df['target'] = train_aug_df['age'].map(class_labels_reassign)
test_df['target'] = test_df['age'].map(class_labels_reassign)


# Converting the filenames and target class labels into lists for augmented train and test datasets.

train_aug_filenames_list = list(train_aug_df['filename'])
train_aug_labels_list = list(train_aug_df['target'])

test_filenames_list = list(test_df['filename'])
test_labels_list = list(test_df['target'])


# Creating tensorflow constants of filenames and labels for augmented train and test datasets from the lists defined above.

train_aug_filenames_tensor = tf.constant(train_aug_filenames_list)
train_aug_labels_tensor = tf.constant(train_aug_labels_list)

test_filenames_tensor = tf.constant(test_filenames_list)
test_labels_tensor = tf.constant(test_labels_list)


# Defining a function to read the image, decode the image from given tensor and one-hot encode the image label class.
# Changing the channels para in tf.io.decode_jpeg from 3 to 1 changes the output images from RGB coloured to grayscale.

num_classes = 4

def _parse_function(filename, label):
    
    image_string = tf.io.read_file(filename)
    image_decoded = tf.io.decode_jpeg(image_string, channels=1)    # channels=1 to convert to grayscale, channels=3 to convert to RGB.
    # image_resized = tf.image.resize(image_decoded, [200, 200])
    label = tf.one_hot(label, num_classes)

    return image_decoded, label



# Getting the dataset ready for the neural network.
# Using the tensor vectors defined above, accessing the images in the dataset and passing them through the function defined above.

train_aug_dataset = tf.data.Dataset.from_tensor_slices((train_aug_filenames_tensor, train_aug_labels_tensor))
train_aug_dataset = train_aug_dataset.map(_parse_function)
train_aug_dataset = train_aug_dataset.batch(512)    # Same as batch_size hyperparameter in model.fit() below.

test_dataset = tf.data.Dataset.from_tensor_slices((test_filenames_tensor, test_labels_tensor))
test_dataset = test_dataset.map(_parse_function)
test_dataset = test_dataset.batch(512)    # Same as batch_size hyperparameter in model.fit() below.


# Defining the architecture of the sequential neural network.

model = Sequential()
inp = Input(shape=(img_height, img_width, 1))

# Input layer with 32 filters, followed by an AveragePooling2D layer.
cnn = Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=(200, 200, 1))
model.add(cnn)    # 3rd dim = 1 for grayscale images.
pooling = AveragePooling2D(pool_size=(2,2))
model.add(pooling)

# Three Conv2D layers with filters increasing by a factor of 2 for every successive Conv2D layer.
cnn = Conv2D(filters=64, kernel_size=3, activation='relu')
model.add(cnn)
pooling = AveragePooling2D(pool_size=(2,2))
model.add(pooling)

cnn = Conv2D(filters=128, kernel_size=3, activation='relu')
model.add(cnn)
pooling = AveragePooling2D(pool_size=(2,2))
model.add(pooling)

cnn = Conv2D(filters=256, kernel_size=3, activation='relu')
model.add(cnn)
pooling = AveragePooling2D(pool_size=(2,2))
model.add(pooling)

# A GlobalAveragePooling2D layer before going into Dense layers below.
# GlobalAveragePooling2D layer gives no. of outputs equal to no. of filters in last Conv2D layer above (256).
model.add(GlobalAveragePooling2D())

# One Dense layer with 132 nodes so as to taper down the no. of nodes from no. of outputs of GlobalAveragePooling2D layer above towards no. of nodes in output layer below (7).
fc1 = Dense(units=132, activation='relu')
model.add(fc1)

# Output layer with 7 nodes (equal to the no. of classes).
fc2 = Dense(num_classes, activation='softmax')
model.add(fc2)

model.summary()


# Compiling the above created CNN architecture.

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# Creating a TensorBoard callback object and saving it at the desired location.

# tensorboard = TensorBoard(log_dir=f"/content/drive/My Drive/1_LiveProjects/Project5_AgeGenderEmotion_Detection/1.1_age_input_output/output/cnn_logs")


# Creating a ModelCheckpoint callback object to save the model according to the value of val_accuracy.

# checkpoint = ModelCheckpoint(filepath=f"/content/drive/My Drive/1_LiveProjects/Project5_AgeGenderEmotion_Detection/1.1_age_input_output/output/cnn_logs/age_model_checkpoint.h5",
#                              monitor='val_accuracy',
#                              save_best_only=True,
#                              save_weights_only=False,
#                              verbose=1
#                             )


# Fitting the above created CNN model.

history = model.fit(train_aug_dataset,
                                  batch_size=512,
                                  validation_data=test_dataset,
                                  epochs=60,
                                  shuffle=False    # shuffle=False to reduce randomness and increase reproducibility
                                 )


# Checking the train and test loss and accuracy values from the neural network above.

his = history
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
# train_loss = model_history.history['loss']
# test_loss = model_history.history['val_loss']
# train_accuracy = model_history.history['accuracy']
# test_accuracy = model_history.history['val_accuracy']


# Saving the model as a h5 file for possible use later.

model.save('age_detection_model.h5')