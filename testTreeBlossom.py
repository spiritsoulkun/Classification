import numpy as np
import os
import pathlib
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.ops.gen_math_ops import imag
#set data directory
#data_dir = pathlib.path("C:\\Users\\Administrator\\Documents\\data")
#data_dir = pathlib.path("C:/Users/Administrator/Documents/data")
data_dir = pathlib.Path(r"C:\Users\Administrator\Documents\data")
print("training: ")
print('Total # of daisy images: ', len(list(data_dir.glob('training/daisy/*.jpg'))))
print('Total # of dandelion images: ', len(list(data_dir.glob('training/dandelion/*.jpg'))))
print('Total # of roses images: ', len(list(data_dir.glob('training/roses/*.jpg'))))
print('Total # of sunflowers images: ', len(list(data_dir.glob('training/sunflowers/*.jpg'))))
print('Total # of tulips images: ', len(list(data_dir.glob('training/tulips/*.jpg'))))
print('Testing: ')
print('Total # of daisy images: ', len(list(data_dir.glob('testing/daisy/*.jpg'))))
print('Total # of dandelion images: ', len(list(data_dir.glob('testing/dandelion/*.jpg'))))
print('Total # of roses images: ', len(list(data_dir.glob('testing/roses/*.jpg'))))
print('Total # of sunflowers images: ', len(list(data_dir.glob('testing/sunflowers/*.jpg'))))
print('Total # of tulips images: ', len(list(data_dir.glob('testing/tulips/*.jpg'))))

batch_size = 32
img_height = 128
img_width = 128

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(data_dir, 'training'),
    validation_split=0.2,
    subset='training',
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(data_dir, 'training'),
    validation_split=0.2,
    subset='validation',
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

#display all class names -- this step is optional
class_names = train_ds.class_names
print(class_names)

# Here are the first 9 images from the training dataset
# -- this step is optional
plt.figure(figsize=(10,10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i+1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")

num_classes = 5
model =  keras.Sequential([
    layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
    layers.Conv2D(8, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
loss=tf.keras.losses.SparseCategoricalCrossentropy(),
metrics=['accuracy'])
model.summary()

#training
import time
start_time =  time.time()

epochs = 10
history = model.fit(
    train_ds,
    validation_data = val_ds,
    epochs = epochs
)
training_time = (time.time() - start_time)/60
print("---training time: %s minutes---" % training_time)

# save model
# model_path = pathlib.Path(r'C:\Users\Administrator\Documents\data\tensorflow')
# model.save(os.path.join(model_path, 'flower_recognition_model.h5'))
# load model
# flower_model = keras.models.load_model(os.path.join(model_path, 'flower_recognition_model.h5'))

