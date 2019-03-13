import ast
from tkinter import *

import numpy as np
from PIL import ImageTk, Image

from threedee_tools.renderer import Renderer

master = Tk()

image = None
photo = None


# [0.9, 1.0, 0.9, 1.0, 0.8, 1.0, 1.0, 1.0, 1.0, 0.7, 1.0, 1.0, 1.0, 1.0, 1.0, 0.8, 1.0, 1.0, 1.0, 1.0, 1.0, 0.8, 0.7, 0.8, 1.0, 0.7, 0.7, 1.0, 0.9, 1.0, 1.0, 0.9, 0.7, 0.9, 0.8, 1.0, 0.8, 1.0, 0.8, 1.0, 1.0, 0.8, 1.0, 1.0, 0.8, 1.0, 0.9, 1.0, 0.9, 1.0, 0.9, 1.0, 1.0, 1.0, 1.0, 0.7, 0.7, 0.8, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.8, 1.0, 0.8, 1.0, 1.0, 0.9, 1.0, 0.7, 1.0, 0.8, 1.0, 0.9, 0.9, 1.0, 0.9, 0.7, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.7, 1.0, 0.9, 1.0, 0.7, 1.0, 0.7, 1.0, 1.0, 0.9, 0.9, 1.0, 1.0, 0.9, 1.0, 1.0, 1.0, 1.0, 0.8, 0.8, 1.0, 1.0, 1.0, 0.8, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9, 0.7, 0.7, 1.0, 0.7, 0.9, 0.9, 1.0, 1.0, 0.9, 1.0, 0.9, 1.0, 0.9, 1.0, 0.7, 1.0, 1.0, 1.0, 1.0, 0.9, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9, 1.0, 1.0, 1.0]

class popupWindow(object):
    def __init__(self, root):
        top = self.top = Toplevel(root)
        self.l = Label(top, text="Enter Config")
        self.l.pack()
        self.e = Entry(top)
        self.e.pack()
        self.b = Button(top, text='Ok', command=self.cleanup)
        self.b.pack()

    def cleanup(self):
        self.value = self.e.get()
        self.top.destroy()


class ShapeConfig():
    def __init__(self):
        self.rotation_speed = 1  # higher is faster, positive is right
        self.cam_z = -2

        self.values = []
        for x in range(6):
            for r in range(30):
                if x == 5 and r > 9:
                    break
                Label(text=r + 1 + (x * 30), relief=RIDGE, width=15).grid(row=r, column=0 + (x * 2))
                s = Scale(master, from_=0., to=1., resolution=0.1, orient=HORIZONTAL)
                s.set(1)
                s.grid(row=r, column=1 + (x * 2))  # length=10,
                self.values.append(s)

        Button(master, text='max', command=self.max).grid(row=10, column=10, columnspan=2)
        Button(master, text='randomize', command=self.randomize).grid(row=11, column=10, columnspan=2)
        Button(master, text='rotate right', command=self.right).grid(row=12, column=10, columnspan=2)
        Button(master, text='rotate left', command=self.left).grid(row=13, column=10, columnspan=2)
        Button(master, text='cam up', command=self.cam_up).grid(row=14, column=10, columnspan=2)
        Button(master, text='cam down', command=self.cam_down).grid(row=15, column=10, columnspan=2)
        Button(master, text='print config', command=self.print_config).grid(row=16, column=10, columnspan=2)
        self.b = Button(master, text="enter values", command=self.popup)
        self.b.grid(row=17, column=10, columnspan=2)

        width = 512
        height = 512

        self.renderer = Renderer(width, height)

        self.image = Image.fromarray(np.zeros((width, height), dtype=np.uint8))

        self.canvas = Canvas(master, height=height, width=width)
        self.canvas.grid(row=0, column=12, rowspan=30)
        # image = image.resize((basewidth, hsize), PIL.Image.ANTIALIAS)
        self.photo = ImageTk.PhotoImage(self.image)
        self.photo_holder = self.canvas.create_image(
            width - (self.image.size[0] / 2),
            height - (self.image.size[1] / 2), image=self.photo)

        self.rot = 0

        self.render()

    def popup(self):
        self.w = popupWindow(master)
        self.b["state"] = "disabled"
        master.wait_window(self.w.top)
        self.b["state"] = "normal"
        x = ast.literal_eval(self.entryValue())
        self.set_values(x)

    def entryValue(self):
        return self.w.value

    def set_values(self, vs):
        if len(vs) != 160:
            print("ERROR: length of inputs should be 160, found:", len(vs))
            return

        for i in range(160):
            self.values[i].set(vs[i])

    def max(self):
        self.set_values([1.] * 160)

    def randomize(self):
        self.set_values(np.random.uniform(.5, 1, 160).tolist())

    def right(self):
        if self.rotation_speed < 5:
            self.rotation_speed += 1
        if self.rotation_speed == 1:
            self.render()

    def left(self):
        if self.rotation_speed > -5:
            self.rotation_speed -= 1
        if self.rotation_speed == -1:
            self.render()
        print(self.rotation_speed)

    def render(self):
        self.image = self.renderer.render(self.get_values(), np.array((0, self.rot, 0)))
        self.photo = ImageTk.PhotoImage(self.image)
        self.canvas.itemconfig(self.photo_holder, image=self.photo)

        self.rot += 0.1 * np.sign(self.rotation_speed) / (2 * np.pi)

        if self.rotation_speed != 0:
            master.after(25 * (6 - abs(self.rotation_speed)), self.render)

    def get_values(self):
        v = [m.get() for m in self.values]
        return v

    def update_img(self, event):
        pass

    def print_config(self):
        print(self.get_values())

    def cam_up(self):
        self.cam_z += 1

    def cam_down(self):
        self.cam_z -= 1

    def step1(self):
        self.image = Image.open("ball.gif")
        self.photo = ImageTk.PhotoImage(self.image)
        self.canvas.itemconfig(self.photo_holder, image=self.photo)


app = ShapeConfig()
master.mainloop()
