import tkinter as tk
from tkinter import BOTH, YES, simpledialog

import PIL
import PIL.Image
import PIL.ImageDraw


class DrawingClassifier:
    CANVAS_WIDTH = 500
    CANVAS_HEIGHT = 500
    N_CLASSES = 3
    BRUSH_WIDTH = 10
    WHITE = (255, 255, 255)

    def __init__(self, root=tk.Tk(), root_title="Drawing Classifier"):
        self.root = root
        self.root.title(root_title)

        self.classes = {}
        self.btns_for_classes = {}

        self.counters = [0] * self.N_CLASSES

        self.modal = None
        self.image1, self.draw = None, None

        self.init_gui()

    def init_gui(self):
        self.canvas = tk.Canvas(self.root, width=self.CANVAS_WIDTH - 10, height=self.CANVAS_HEIGHT - 10, bg="red")
        self.canvas.pack(expand=YES, fill=BOTH)
        self.canvas.bind("<B1-Motion>", self.paint)

        self.image1 = PIL.Image.new("RGB", (self.CANVAS_WIDTH, self.CANVAS_HEIGHT), self.WHITE)
        self.draw = PIL.ImageDraw.Draw(self.image1)

        for i in range(self.N_CLASSES):
            self.classes[i] = simpledialog.askstring(
                "Classname", f"Enter the name of the {i + 1} class:", parent=self.root
            )

        for i in range(self.N_CLASSES):
            self.btns_for_classes[i] = tk.Button(
                self.root,
                text=self.classes[i],
                width=50,
                command=lambda index=i: self.save_for_class(index),
            )
            self.btns_for_classes[i].pack(anchor=tk.CENTER, expand=True)

        # self.root.protocol("WM_DELETE_WINDOW", self.on_delete)
        self.root.attributes("-topmost", True)
        self.root.mainloop()

    def paint(self, event):
        x1, y1 = (event.x - 1), (event.y - 1)
        x2, y2 = (event.x + 1), (event.y + 1)
        self.canvas.create_rectangle(x1, y1, x2, y2, fill="black", width=self.BRUSH_WIDTH)
        self.draw.rectangle(
            [x1, y2, x2 + self.BRUSH_WIDTH, y2 + self.BRUSH_WIDTH], fill="black", width=self.BRUSH_WIDTH
        )

    def save_for_class(self, index):
        print(f"Save for class: {self.classes[index]}")

    def on_delete(self):
        pass


DrawingClassifier()
