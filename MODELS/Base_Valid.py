from sklearn.model_selection import train_test_split
import tensorflow as tf

from tensorflow.keras import datasets, layers, models, regularizers, callbacks
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical


(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

train_images, valid_images, train_labels, valid_labels = train_test_split(train_images,train_labels, train_size=0.8)

# Normalize pixel values to be between 0 and 1
train_images, valid_images, test_images = train_images / 255.0, valid_images / 255.0, test_images / 255.0

train_labels = to_categorical(train_labels)
valid_labels = to_categorical(valid_labels)
test_labels = to_categorical(test_labels)

model = models.Sequential()

model.add(layers.Conv2D(64, (3,3), padding='same', activation='relu', input_shape = (32, 32, 3)))
model.add(layers.Conv2D(64, (3,3), padding='same', activation='relu',))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Dropout(0.1))

model.add(layers.Conv2D(128, (3,3), padding='same', activation='relu'))
model.add(layers.Conv2D(128, (3,3), padding='same', activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Dropout(0.1))

model.add(layers.Conv2D(256, (3,3), padding='same', activation='relu'))
model.add(layers.Conv2D(256, (3,3), padding='same', activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Dropout(0.1))


model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dropout(0.1))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dropout(0.1))

model.add(layers.Dense(10))

model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


tensorboard = callbacks.TensorBoard(log_dir="logs/Base_Valid")

model.fit(train_images, train_labels, validation_data=(valid_images, valid_labels), epochs=100, callbacks=[tensorboard])

model.evaluate(test_images, test_labels)

model.save('saved_model/Base_Valid')