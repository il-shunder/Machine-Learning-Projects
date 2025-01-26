import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from keras import datasets, layers, losses, models

MODEL_PATH = "image_classifier.keras"
CLASS_NAMES = ["Plane", "Car", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck"]
MAX_PIXEL_VALUE = 255

try:
    model = models.load_model(MODEL_PATH)
except ValueError:
    (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
    x_train = x_train / MAX_PIXEL_VALUE
    x_test = x_test / MAX_PIXEL_VALUE

    model = models.Sequential(
        [
            layers.Input(shape=(32, 32, 3)),
            layers.Conv2D(32, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(64, activation="relu"),
            layers.Dense(10, activation="linear"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss=losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    model.fit(x_train, y_train, epochs=10, verbose=2)

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)

    print(f"\nTest loss: {test_loss}")
    print(f"\nTest accuracy: {test_acc}")

    model.save(MODEL_PATH)
finally:
    if model:
        img = cv.imread("test_images/dog.jpg")
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        prediction = model.predict(np.array([img]) / MAX_PIXEL_VALUE)
        index = np.argmax(prediction)

        print(f"Predicted class: {CLASS_NAMES[index]}")

        plt.imshow(img, cmap=plt.cm.binary)
        plt.show()
