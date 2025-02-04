import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import layers, losses, models

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=-1)
x_test = tf.keras.utils.normalize(x_test, axis=-1)

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

model.fit(x_train, y_train, epochs=5)
