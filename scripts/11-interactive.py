import ast
from tkinter import *

import moderngl
import numpy as np
import PIL
from PIL import ImageTk, Image
from pyrr import Matrix44

from threedee_tools.shaders import VERTEX_SHADER_NORMAL, FARGMENT_SHADER_LIGHT_COLOR
from threedee_tools.utils import sphere_vertices, CUBE_FACE, get_unique_vertices, scale_vertices, FACE_COLORS, \
    scale_vertices_dry

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

        self.ctx = moderngl.create_standalone_context()
        self.prog = self.ctx.program(vertex_shader=VERTEX_SHADER_NORMAL, fragment_shader=FARGMENT_SHADER_LIGHT_COLOR)

        self.verts_base, self.faces_base = sphere_vertices(CUBE_FACE, 5)

        self.unique_verts = get_unique_vertices(self.verts_base)
        self.iiis = scale_vertices_dry(self.faces_base, self.unique_verts)

        self.ctx.enable(moderngl.DEPTH_TEST)
        self.fbo = self.ctx.simple_framebuffer((width, height))
        self.fbo.use()

        self.proj = Matrix44.perspective_projection(45.0, width / height, 0.1, 1000.0)

        # light position (will stay in place, independent of object/cam rotation)
        self.base_light = np.array((10, 10, -10))

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
            print("ERROR: length of inputs should be 160, found:", len(x))
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
        faces = [f.copy() for f in self.faces_base]
        scale_vertices(faces, self.get_values(), iiis=self.iiis)

        vertex_buffers = []
        for face in faces:
            face.calculate_normals()
            verts = np.dstack([face.x(), face.y(), face.z(), face.nx(), face.ny(), face.nz()])
            vbo = self.ctx.buffer(verts.astype('f4').tobytes())
            vao = self.ctx.simple_vertex_array(self.prog, vbo, 'in_vert', 'in_norm')
            vertex_buffers.append(vao)

        self.fbo.clear(0.0, 0.0, 0.0, 0.0)

        lookat = Matrix44.look_at(
            (2, 2, self.cam_z),  # eye / camera position
            (0.0, 0.0, 0.0),  # lookat
            (0.0, 0.0, 1.0),  # camera up vector
        )

        rotate = np.array(Matrix44.from_z_rotation(self.rot))
        self.prog['Lights'].value = tuple(np.matmul(rotate[:3, :3], self.base_light).reshape(1, -1)[0])
        self.prog['Mvp'].write((self.proj * lookat * rotate).astype('f4').tobytes())

        for vb, color in zip(vertex_buffers, FACE_COLORS):
            self.prog['Color'].value = color
            vb.render(moderngl.TRIANGLES)

        self.image = Image.frombytes('RGB', self.fbo.size, self.fbo.read(), 'raw', 'RGB', 0, -1)
        self.photo = ImageTk.PhotoImage(self.image)
        self.canvas.itemconfig(self.photo_holder, image=self.photo)

        self.rot += 0.1 * np.sign(self.rotation_speed)

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
