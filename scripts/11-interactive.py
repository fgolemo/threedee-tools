from tkinter import *
import numpy as np
import PIL
from PIL import ImageTk, Image

master = Tk()

image = None
photo = None


class ShapeConfig():
    def __init__(self):
        self.values = []
        for x in range(6):
            for r in range(30):
                if x == 5 and r > 9:
                    break
                Label(text=r + 1 + (x * 30), relief=RIDGE, width=15).grid(row=r, column=0 + (x * 2))
                s = Scale(master, from_=0., to=1., resolution=0.1, orient=HORIZONTAL, command=self.update_img)
                s.set(1)
                s.grid(row=r, column=1 + (x * 2))  # length=10,
                self.values.append(s)

        Button(master, text='max', command=self.step1).grid(row=10, column=10, columnspan=2)
        Button(master, text='randomize', command=self.step1).grid(row=11, column=10, columnspan=2)
        Button(master, text='rotate right', command=self.step1).grid(row=12, column=10, columnspan=2)
        Button(master, text='rotate left', command=self.step1).grid(row=13, column=10, columnspan=2)
        Button(master, text='print config', command=self.print_config).grid(row=14, column=10, columnspan=2)

        width = 512
        height = 512
        self.image = Image.fromarray(np.zeros((width, height), dtype=np.uint8))

        self.canvas = Canvas(master, height=height, width=width)
        self.canvas.grid(row=0, column=12, rowspan=30)
        # image = image.resize((basewidth, hsize), PIL.Image.ANTIALIAS)
        self.photo = ImageTk.PhotoImage(self.image)
        self.photo_holder = self.canvas.create_image(
            width - (self.image.size[0] / 2),
            height - (self.image.size[1] / 2), image=self.photo)

    def get_values(self):
        v = [m.get() for m in self.values]
        return v

    def update_img(self, event):
        pass

    def print_config(self):
        print(self.get_values())

    def step1(self):
        self.image = Image.open("ball.gif")
        self.photo = ImageTk.PhotoImage(self.image)
        self.canvas.itemconfig(self.photo_holder, image=self.photo)


app = ShapeConfig()
master.mainloop()

