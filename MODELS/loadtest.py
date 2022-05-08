import tensorflow as tf


(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

train_labels = tf.keras.utils.to_categorical(train_labels)
test_labels = tf.keras.utils.to_categorical(test_labels)

new_model = tf.keras.models.load_model('saved_model/my_model')

loss, acc = new_model.evaluate(test_images, test_labels, verbose=2)
print('Restored model, accuracy@ {:5.2f}%'.format(100*acc))

print(new_model.predict(test_images).shape)
