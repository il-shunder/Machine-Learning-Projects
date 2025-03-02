import os
import tkinter as tk
from tkinter import simpledialog

import PIL
import PIL.Image
import PIL.ImageDraw


class DrawingClassifier:
    CANVAS_WIDTH = 500
    CANVAS_HEIGHT = 500
    N_CLASSES = 3
    BRUSH_WIDTH = 15
    TEMP_IMAGE = "image.png"
    TRAIN_IMAGE_SIZE = (50, 50)

    def __init__(self, root=tk.Tk(), root_title="Drawing Classifier"):
        self.root = root
        self.root.title(root_title)

        self.classes = {}
        self.btns_for_classes = {}

        self.counters = [0] * self.N_CLASSES

        self.modal = None
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

        change_btn = tk.Button(btn_frame, text="Change Model", command=self.change_model)
        change_btn.grid(row=1, column=0, sticky=tk.NSEW)

        train_btn = tk.Button(btn_frame, text="Train Model", command=self.train_model)
        train_btn.grid(row=1, column=1, sticky=tk.NSEW)

        clear_btn = tk.Button(btn_frame, text="Clear", command=self.clear)
        clear_btn.grid(row=1, column=2, sticky=tk.NSEW)

        predict_btn = tk.Button(btn_frame, text="Predict", command=self.predict)
        predict_btn.grid(row=2, columnspan=3, sticky=tk.NSEW)

        self.status_label = tk.Label(btn_frame, text=f"Current Model: {type(self.modal).__name__}")
        self.status_label.config(font=("Arial", 20))
        self.status_label.grid(row=4, column=1, sticky=tk.NSEW)

        # self.root.protocol("WM_DELETE_WINDOW", self.on_delete)
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

        self.image.save(self.TEMP_IMAGE)

        filepath = f"{index}/image{self.counters[index]}.png"
        img = PIL.Image.open(self.TEMP_IMAGE)
        img.thumbnail(self.TRAIN_IMAGE_SIZE, PIL.Image.Resampling.LANCZOS)
        img.save(filepath)

        if os.path.isfile(self.TEMP_IMAGE):
            os.remove(self.TEMP_IMAGE)

        self.counters[index] += 1
        self.clear()

    def clear(self):
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, 1000, 1000], fill="white", outline="white")

    def train_model(self):
        pass

    def change_model(self):
        pass

    def predict(self):
        pass

    def on_delete(self):
        pass


DrawingClassifier()
