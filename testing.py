import numpy as np
import os
import pathlib
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.preprocessing.image import img_to_array
from tensorflow.python.ops.gen_math_ops import imag
data_dir = pathlib.Path(r"C:\Users\Administrator\Documents\data")
# load testing data
img_height = 128
img_width = 128
batch_size = 32
num_of_test_samples = 470
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(data_dir, 'testing'),
    image_size=(img_height, img_width),
    batch_size=num_of_test_samples
)

# load the model
model_path = pathlib.Path(r'C:\Users\Administrator\Documents\data\tensorflow')
flower_model = keras.models.load_model(os.path.join(model_path, 'flower_recognition_model.h5'))

# evaluate the model
accuracy = flower_model.evaluate(test_ds, verbose=1)
print('Loss on the test set: %.2f' % (accuracy[0]*100))
print('Accuracy on the test set: %.2f' % (accuracy[1]*100))

#specify test image filename

test_img_fn = os.path.join(data_dir, 'testing/roses/118974357_0faa23cce9_n.jpg')

# read image

img = keras.preprocessing.image.load_img(
    test_img_fn, target_size=(img_height, img_width)
)

#convert image into array
img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) #Create a batch

#apply the model to predict flower name
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(data_dir, 'training'),
    validation_split=0.2,
    subset='training',
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)
class_names = train_ds.class_names
prob =  flower_model.predict(img_array)
predicted_class = class_names[np.argmax(prob)]

#display the result
print('Predicted class: ', predicted_class, '(Probability = ', np.max(prob),')')