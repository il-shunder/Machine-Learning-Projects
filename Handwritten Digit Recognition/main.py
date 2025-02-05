import os

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import layers, losses, models

MODEL_PATH = "digit_recognition.keras"
DIGIT_PATH = "digits/digit"
DIGIT_FILE_EXTENSION = ".png"

try:
    model = models.load_model(MODEL_PATH)
except ValueError:
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = tf.keras.utils.normalize(x_train, axis=1)
    x_test = tf.keras.utils.normalize(x_test, axis=1)

    model = models.Sequential(
        [
            layers.Input((28, 28, 1)),
            layers.Conv2D(32, (3, 3), activation="relu"),
            layers.MaxPool2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPool2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPool2D((2, 2)),
            layers.Flatten(),
            layers.Dense(64, activation="relu"),
            layers.Dense(10, activation="linear"),
        ]
    )

    model.compile(optimizer="adam", loss=losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])
    model.fit(x_train, y_train, epochs=10)
    model.save(MODEL_PATH)
finally:
    if model:
        file_num = 1
        while os.path.isfile(DIGIT_PATH + str(file_num) + DIGIT_FILE_EXTENSION):
            try:
                image = cv.imread(DIGIT_PATH + str(file_num) + DIGIT_FILE_EXTENSION)[:, :, 0]
                image = np.invert(np.array([image]))
                image = tf.keras.utils.normalize(image, axis=1)
                prediction = model.predict(image, verbose=0)
                print(f"Predicted number is: {np.argmax(prediction)}")
                plt.imshow(image[0], cmap=plt.cm.binary)
                plt.show()
            except:
                print("An error occurred :(")
            finally:
                file_num += 1
