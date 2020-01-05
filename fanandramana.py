# This section take images from plante disease files and imput them_
# to the postgres SQL
##################################################################################################################################################
import os
import cv2 as cv
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer
# %autoindent

SZ = 128 

DIR_IMAGE_DATA = '/media/tanjona/Disque local D/BOSS/PlanteDisease'

CATEGORIES = os.listdir(DIR_IMAGE_DATA)

def create_train_data():
    label_list = []
    images = []
    max_class_size = 800
    for category in CATEGORIES:
        path = os.path.join(DIR_IMAGE_DATA, category)
        images_per_class = os.listdir(path)
        images_per_class = images_per_class[:max_class_size]
        for img in images_per_class:
            img_array = cv.imread(os.path.join(path, img))
            new_array = cv.resize(img_array, (SZ, SZ))
            images.append(new_array)
            label_list.append(category)

    return images, label_list



images, label_list = create_train_data()
label_list = np.array(label_list)

np_images = np.array(images, dtype=np.float32)/255.0
label_binarizer = LabelBinarizer()
image_labels = label_binarizer.fit_transform(label_list)
pickle.dump(label_binarizer, open('label_transform.pk1', 'wb'))
n_classes = len(label_binarizer.classes_)

x_train, x_test, y_train, y_test = train_test_split(np_images, image_labels, test_size=0.2 )
aug = ImageDataGenerator(
        rotation_range=25,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest")




model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3),input_shape=(128, 128, 3), activation='relu', ))
model.add(tf.keras.layers.MaxPool2D(pool_size=(3, 3)))
model.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Conv2D(128, kernel_size=(3, 3),  activation='relu'))
model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))


model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(len(CATEGORIES), activation='softmax'))

model.summary()

model.compile(
    loss = 'binary_crossentropy',
    optimizer = 'Adam',
    metrics = ['accuracy']
)

history = model.fit_generator(
            aug.flow(x_train, y_train, batch_size=32),
            validation_data=(x_test, y_test),
            steps_per_epoch=len(x_train) // 32,
            epochs=6, verbose=1
            )

model.save('leaf_disease_detection.h5')

loaded_model = tf.keras.models.load_model('leaf_disease_detection.h5')

















