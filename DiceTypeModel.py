import tensorflow as tf
from tensorflow.keras import datasets, models, layers
import matplotlib.pyplot as plot

PATH_TO_TRAIN_DATA = 'DiceType/dice/train'
BATCH_SIZE = 32
IMG_HEIGHT = 128
IMG_WIDTH = 128

#Train data consists of a tuple of image_batches and label_batches
trainData = tf.keras.preprocessing.image_dataset_from_directory(
    PATH_TO_TRAIN_DATA,
    labels='inferred',
    label_mode='int',
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset='training',
    seed=100)

#Create validation data
validationData = tf.keras.preprocessing.image_dataset_from_directory(
    PATH_TO_TRAIN_DATA,
    labels='inferred',
    label_mode='int',
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    validation_split = 0.2,
    subset='validation',
    seed=100)

#The classifications of dice
classNames = trainData.class_names

#This code will show some labeled data to make sure it is has been imported properly
for images, labels in trainData.take(1): #Take one batch from the dataset and test images to see if properly labeled
    for i in range (9):
        ax = plot.subplot(3, 3, i + 1)
        plot.imshow(images[i].numpy().astype("uint8"))
        plot.title(classNames[labels[i]])
        plot.axis("off")
#plot.show()

#Define CNN Model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(6)) #There are 6 classifications in our dataset

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

images, labels = tuple(zip(*trainData))
model.fit(trainData, epochs=6, validation_data = validationData)
model.save('savedModels/DiceTypeModel')

