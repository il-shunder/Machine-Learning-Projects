import tkinter as tk
from tkinter import simpledialog


class DrawingClassifier:
    def __init__(self, root=tk.Tk(), root_title="Drawing Classifier"):
        self.root = root
        self.root_title = root_title

        self.class1, self.class2, self.class3 = None
        self.class1_counter, self.class2_counter, self.class3_counter = None

        self.modal = None

        self.init_gui()

        self.root.attributes("-topmost", True)
        self.root.mainloop()

    def init_gui(self):
        pass


DrawingClassifier()
