import os
import tkinter as tk
from tkinter import simpledialog, ttk

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
    TEMP_IMAGE_PATH = "image.png"
    TRAIN_IMAGE_SIZE = (50, 50)

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
        self.canvas.pack(expand=tk.YES, fill=tk.BOTH)
        self.canvas.bind("<B1-Motion>", self.paint)

        self.image = PIL.Image.new("RGB", (self.CANVAS_WIDTH, self.CANVAS_HEIGHT), (255, 255, 255))
        self.draw = PIL.ImageDraw.Draw(self.image)

        for i in range(self.N_CLASSES):
            self.classes[i] = simpledialog.askstring(
                "Classname", f"Enter the name of the {i + 1} class:", parent=self.root
            )

        btn_frame = tk.Frame(self.root)
        btn_frame.pack(fill=tk.X, side=tk.BOTTOM)

        btn_frame.columnconfigure(0, weight=1)
        btn_frame.columnconfigure(1, weight=1)
        btn_frame.columnconfigure(2, weight=1)

        for i in range(self.N_CLASSES):
            self.btns_for_classes[i] = tk.Button(
                btn_frame,
                text=self.classes[i],
                command=lambda index=i: self.save_image(index),
            )
            self.btns_for_classes[i].grid(row=0, column=i, sticky=tk.NSEW)

        self.model_dropdown = ttk.Combobox(btn_frame, state="readonly", values=self.model_options)
        self.model_dropdown.current(0)
        self.model_dropdown.grid(row=1, column=0, sticky=tk.NSEW)
        self.model_dropdown.bind("<<ComboboxSelected>>", self.change_model)

        train_btn = tk.Button(btn_frame, text="Train Model", command=self.train_model)
        train_btn.grid(row=1, column=1, sticky=tk.NSEW)

        clear_btn = tk.Button(btn_frame, text="Clear", command=self.clear)
        clear_btn.grid(row=1, column=2, sticky=tk.NSEW)

        predict_btn = tk.Button(btn_frame, text="Predict", command=self.predict)
        predict_btn.grid(row=2, columnspan=3, sticky=tk.NSEW)

        self.status_label = tk.Label(btn_frame, text=f"Current Model: {type(self.model).__name__}")
        self.status_label.config(font=("Arial", 20))
        self.status_label.grid(row=4, column=1, sticky=tk.NSEW)

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

        if os.path.isfile(self.TEMP_IMAGE_PATH):
            os.remove(self.TEMP_IMAGE_PATH)

        self.counters[index] += 1
        self.clear()

    def clear(self):
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, 1000, 1000], fill="white", outline="white")

    def train_model(self):
        pass

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
        pass

    def on_delete(self):
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


DrawingClassifier()
