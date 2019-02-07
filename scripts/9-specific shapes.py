import moderngl
import numpy as np
from threedee_tools.utils_3d import sphere_vertices, CUBE_FACE, FACE_COLORS, get_unique_vertices, scale_vertices, \
    cube_vertices
from PIL import Image
from pyrr import Matrix44

ctx = moderngl.create_standalone_context()

prog = ctx.program(
    vertex_shader='''
        #version 330

        uniform mat4 Mvp;

        in vec3 in_vert;
        in vec3 in_norm;

        out vec3 v_vert;
        out vec3 v_norm;

        void main() {
            gl_Position = Mvp * vec4(in_vert, 1.0);
            v_vert = in_vert;
            v_norm = in_norm;
        }
    ''',
    fragment_shader='''
        #version 330

        uniform vec3 Lights;
        uniform vec3 Color;

        in vec3 v_vert;
        in vec3 v_norm;

        out vec4 f_color;

        void main() {
            float lum = clamp(
                dot(
                    normalize(Lights - v_vert), 
                    normalize(v_norm)
                ),
                0.0, 
                1.0) * 0.6 + 0.4;
            f_color = vec4(Color * lum, 1.);
        }
    ''',
)
prog['Lights'].value = (100, 100, 100)
print(prog._members.items())

# random_scaling = np.random.uniform(0, 1, 160)
sphere = np.ones((160)) * 0.7
cube = np.ones((160))
cube[-40:] = .7

verts, faces = cube_vertices()

vertex_buffers = []
for face in faces:
    face.calculate_normals()
    verts = np.dstack([face.x(), face.y(), face.z(), face.nx(), face.ny(), face.nz()])
    vbo = ctx.buffer(verts.astype('f4').tobytes())
    vao = ctx.simple_vertex_array(prog, vbo, 'in_vert', 'in_norm')
    vertex_buffers.append(vao)

width = 512
height = 512

ctx.enable(moderngl.DEPTH_TEST)
fbo = ctx.simple_framebuffer((width, height))
fbo.use()

proj = Matrix44.perspective_projection(45.0, width / height, 0.1, 1000.0)
lookat = Matrix44.look_at(
    (2, 2, 1),  # eye / camera position
    (0.0, 0.0, 0.0),  # lookat
    (0.0, 0.0, 1.0),  # camera up vector
)

import matplotlib.pyplot as plt

f, axarr = plt.subplots(1, 13, sharex=True, sharey=True, figsize=(20, 2))

base_light = np.array((100, 100, 100))

for rot in np.arange(0, 6.1, 0.5):
    fbo.clear(0.0, 0.0, 0.0, 0.0)
    rotate = np.array(Matrix44.from_z_rotation(rot))

    # this keep the light in place
    prog['Lights'].value = tuple(np.matmul(rotate[:3, :3], base_light).reshape(1, -1)[0])
    # prog['Lights'].value = (10, 10, 10)

    prog['Mvp'].write((proj * lookat * rotate).astype('f4').tobytes())

    for vb, color in zip(vertex_buffers, FACE_COLORS):
        prog['Color'].value = color
        vb.render(moderngl.TRIANGLES)

    img = Image.frombytes('RGB', fbo.size, fbo.read(), 'raw', 'RGB', 0, -1)

    # plt.subplot(171 + (rot * 2))
    axarr[int(rot * 2)].imshow(img)

plt.tight_layout()
plt.show()
