#!/usr/bin/env python
# coding: utf-8


import tensorflow as tf
import matplotlib.pyplot as plt
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from keras.layers.core import Dense, Flatten, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D, SeparableConv2D
import keras.models

t = ImageDataGenerator(rescale=1/255, validation_split=0.2)

train = t.flow_from_directory('starter/', target_size= (128,128), batch_size = 3, class_mode='sparse', subset='training' )
test = t.flow_from_directory('starter/', target_size= (128,128), batch_size = 3, class_mode='sparse', subset='validation' )


for _ in range(5):
    img, label = train.next()
    plt.imshow(img[0])
    plt.show()


print(test.labels)


model = keras.models.Sequential()
model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128,3)))
model.add(keras.layers.MaxPooling2D((2, 2)))
model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(keras.layers.MaxPooling2D((2, 2)))
model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(keras.layers.GlobalAveragePooling2D(data_format='channels_last'))
model.add(keras.layers.Dense(6, activation='softmax'))


model.summary()


optimizer = keras.optimizers.Adam(lr=0.0001)
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


foo = model.fit(train, shuffle = True, epochs=20, validation_data = test)


model.save('model/')