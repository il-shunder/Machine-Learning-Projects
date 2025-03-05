import math
import os
import random
import tkinter as tk
from tkinter import simpledialog, ttk

import cv2 as cv
import numpy as np
import PIL
import PIL.Image
import PIL.ImageDraw
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier


class DrawingClassifier:
    CANVAS_WIDTH = 500
    CANVAS_HEIGHT = 500
    N_CLASSES = 3
    BRUSH_WIDTH = 15
    AUGMENTATIONS_PER_IMAGE = 20
    TEMP_IMAGE_PATH = "image.png"
    TRAIN_IMAGE_SIZE = (50, 50)
    BG_COLOR = (255, 255, 255)

    def __init__(self, root=tk.Tk(), root_title="Drawing Classifier"):
        self.root = root
        self.root.title(root_title)

        self.classes = {}
        self.btns_for_classes = {}

        self.counters = [0] * self.N_CLASSES

        self.model = LinearSVC()
        self.model_options = [
            type(self.model).__name__,
            type(RandomForestClassifier()).__name__,
            type(LogisticRegression()).__name__,
            type(GaussianNB()).__name__,
            type(KNeighborsClassifier()).__name__,
            type(DecisionTreeClassifier()).__name__,
        ]
        self.image, self.draw = None, None

        self.init_gui()

    def init_gui(self):
        self.canvas = tk.Canvas(self.root, width=self.CANVAS_WIDTH - 10, height=self.CANVAS_HEIGHT - 10, bg="white")
        self.canvas.pack()
        self.canvas.bind("<B1-Motion>", self.paint)

        self.image = PIL.Image.new("RGB", (self.CANVAS_WIDTH, self.CANVAS_HEIGHT), self.BG_COLOR)
        self.draw = PIL.ImageDraw.Draw(self.image)

        for i in range(self.N_CLASSES):
            self.classes[i] = simpledialog.askstring(
                "Classname", f"Enter the name of the {i + 1} class:", parent=self.root
            )

        btn_frame = tk.Frame(self.root)
        btn_frame.pack(fill=tk.X, side=tk.BOTTOM)

        btn_frame.grid_columnconfigure(0, weight=1, uniform="buttons")
        btn_frame.grid_columnconfigure(1, weight=1, uniform="buttons")
        btn_frame.grid_columnconfigure(2, weight=1, uniform="buttons")

        for i in range(self.N_CLASSES):
            self.btns_for_classes[i] = tk.Button(
                btn_frame,
                text=self.classes[i],
                command=lambda index=i: self.save_image(index),
            )
            self.btns_for_classes[i].grid(row=0, column=i, sticky=tk.NSEW)

        augmentation_btn = tk.Button(btn_frame, text="Data Augmentation", command=self.data_augmentation)
        augmentation_btn.grid(row=1, columnspan=3, sticky=tk.NSEW)

        self.model_dropdown = ttk.Combobox(btn_frame, state="readonly", values=self.model_options)
        self.model_dropdown.current(0)
        self.model_dropdown.grid(row=2, column=0, sticky=tk.NSEW)
        self.model_dropdown.bind("<<ComboboxSelected>>", self.change_model)

        train_btn = tk.Button(btn_frame, text="Train Model", command=self.train_model)
        train_btn.grid(row=2, column=1, sticky=tk.NSEW)

        clear_btn = tk.Button(btn_frame, text="Clear", command=self.clear)
        clear_btn.grid(row=2, column=2, sticky=tk.NSEW)

        predict_btn = tk.Button(btn_frame, text="Predict", command=self.predict)
        predict_btn.grid(row=3, columnspan=3, sticky=tk.NSEW)

        self.predicted_class = tk.Label(btn_frame, text="There is no predicted class")
        self.predicted_class.config(font=("Arial", 20))
        self.predicted_class.grid(row=4, columnspan=3, sticky=tk.NSEW)

        self.root.protocol("WM_DELETE_WINDOW", self.on_delete)
        self.root.attributes("-topmost", True)
        self.root.mainloop()

    def paint(self, event):
        x1, y1 = (event.x - 1), (event.y - 1)
        x2, y2 = (event.x + 1), (event.y + 1)
        self.canvas.create_rectangle(x1, y1, x2, y2, fill="black", outline="black", width=self.BRUSH_WIDTH)
        self.draw.rectangle(
            [x1, y2, x2 + self.BRUSH_WIDTH, y2 + self.BRUSH_WIDTH],
            fill="black",
            outline="black",
            width=self.BRUSH_WIDTH,
        )

    def save_image(self, index):
        if not os.path.isdir(str(index)):
            os.mkdir(str(index))

        self.image.save(self.TEMP_IMAGE_PATH)

        filepath = f"{index}/image{self.counters[index]}.png"
        img = PIL.Image.open(self.TEMP_IMAGE_PATH)
        img.thumbnail(self.TRAIN_IMAGE_SIZE, PIL.Image.Resampling.LANCZOS)
        img.save(filepath)

        self.counters[index] += 1
        self.clear()

    def clear(self):
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, 1000, 1000], fill="white", outline="white")

    def train_model(self):
        X_train, y_train = self.get_training_data()
        self.model.fit(X_train, y_train)
        print("Model successfully trained!")

    def get_training_data(self):
        img_list = class_list = np.array([])
        n_training_examples = sum(self.counters)

        if n_training_examples > 0:
            img_shape = self.TRAIN_IMAGE_SIZE[0] * self.TRAIN_IMAGE_SIZE[1]
            for i in range(len(self.counters)):
                for j in range(self.counters[i]):
                    img = cv.imread(f"{i}/image{j}.png")[:, :, 0]
                    img = img.reshape(img_shape)
                    img_list = np.append(img_list, [img])
                    class_list = np.append(class_list, i)
            img_list = img_list.reshape(n_training_examples, img_shape)

        return img_list, class_list

    def change_model(self, dropdown):
        dropdown = self.model_dropdown.get()

        if dropdown == "RandomForestClassifier":
            self.set_model(RandomForestClassifier())
        elif dropdown == "LogisticRegression":
            self.set_model(LogisticRegression())
        elif dropdown == "GaussianNB":
            self.set_model(GaussianNB())
        elif dropdown == "KNeighborsClassifier":
            self.set_model(KNeighborsClassifier())
        elif dropdown == "DecisionTreeClassifier":
            self.set_model(DecisionTreeClassifier())
        else:
            self.set_model(LinearSVC())

    def set_model(self, model):
        self.model = model

    def predict(self):
        self.image.save(self.TEMP_IMAGE_PATH)
        img = PIL.Image.open(self.TEMP_IMAGE_PATH)
        img.thumbnail(self.TRAIN_IMAGE_SIZE, PIL.Image.Resampling.LANCZOS)
        img.save(self.TEMP_IMAGE_PATH)

        img = cv.imread(self.TEMP_IMAGE_PATH)[:, :, 0]
        img = img.reshape(self.TRAIN_IMAGE_SIZE[0] * self.TRAIN_IMAGE_SIZE[1])
        prediction = self.model.predict([img])

        self.predicted_class.config(text=f"Predicted Class: {self.classes[prediction[0]]}")

        self.clear()

    def on_delete(self):
        if os.path.isfile(self.TEMP_IMAGE_PATH):
            os.remove(self.TEMP_IMAGE_PATH)
        self.remove_training_data()
        self.root.destroy()
        exit()

    def remove_training_data(self):
        for dir in range(len(self.counters)):
            dir = str(dir)
            if os.path.isdir(dir):
                for file in os.listdir(dir):
                    file_path = os.path.join(dir, file)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                os.rmdir(dir)

    def data_augmentation(self):
        n_training_examples = sum(self.counters)

        if n_training_examples > 0:
            h_move = math.ceil(self.TRAIN_IMAGE_SIZE[0] / 5)
            v_move = math.ceil(self.TRAIN_IMAGE_SIZE[1] / 5)
            w_zoom = self.TRAIN_IMAGE_SIZE[0] / 2
            h_zoom = self.TRAIN_IMAGE_SIZE[1] / 2

            for i in range(len(self.counters)):
                counter = self.counters[i]
                for j in range(self.counters[i]):
                    for a in range(self.AUGMENTATIONS_PER_IMAGE):
                        filepath = f"{i}/image{counter}.png"
                        img = PIL.Image.open(f"{i}/image{j}.png")
                        img = self.zoom_image(img, w_zoom, h_zoom, random.uniform(1.0, 1.4))
                        img = self.move_image(
                            img, (random.randrange(-h_move, h_move), random.randrange(-v_move, v_move))
                        )
                        img.save(filepath)
                        counter += 1
                self.counters[i] = counter

    def move_image(self, img, translate):
        return img.rotate(0, translate=(translate), fillcolor=self.BG_COLOR)

    def zoom_image(self, img, x, y, zoom):
        w, h = img.size
        zoom2 = zoom * 2
        img = img.crop((x - w / zoom2, y - h / zoom2, x + w / zoom2, y + h / zoom2))
        return img.resize((w, h), PIL.Image.Resampling.LANCZOS)


DrawingClassifier()
