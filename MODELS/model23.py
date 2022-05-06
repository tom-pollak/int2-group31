from doctest import OutputChecker
import tensorflow as tf

from tensorflow.keras import datasets, layers, models, regularizers, callbacks
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical


(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

model = models.Sequential()

model.add(layers.Conv2D(64, (3,3), padding='same', activation='relu', input_shape = (32, 32, 3)))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(64, (3,3), padding='same', activation='relu',))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Dropout(0.2))

model.add(layers.Conv2D(128, (3,3), padding='same', activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(128, (3,3), padding='same', activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Dropout(0.3))

model.add(layers.Conv2D(256, (3,3), padding='same', activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(256, (3,3), padding='same', activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Dropout(0.4))


model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.6))


model.add(layers.Dense(10))

model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

generator = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2, 
    height_shift_range=0.2,
    zoom_range = 0.2,
    shear_range=0.2, 
    channel_shift_range = 0.2,
    featurewise_center = True,
    horizontal_flip=True
)
 
train_generator = generator.flow(train_images, train_labels, 32)
steps_per_epoch = train_images.shape[0] // 32


tensorboard = callbacks.TensorBoard(log_dir="logs/model-23")
 
history = model.fit(train_generator, steps_per_epoch = steps_per_epoch, validation_data=(test_images, test_labels), epochs=800, callbacks=[tensorboard])