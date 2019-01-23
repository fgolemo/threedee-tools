import moderngl
import numpy as np
from threedee_tools.utils import sphere_vertices, CUBE_FACE, FACE_COLORS, get_unique_vertices, scale_vertices
from PIL import Image
from pyrr import Matrix44

ctx = moderngl.create_standalone_context()

prog = ctx.program(
    vertex_shader='''
        #version 330

        uniform mat4 Mvp;

        in vec3 in_vert;
        //in vec3 in_norm;
        //in vec2 in_text;

        out vec3 v_vert;
        //out vec3 v_norm;
        //out vec2 v_text;

        void main() {
            gl_Position = Mvp * vec4(in_vert, 1.0);
            v_vert = in_vert;
            //v_norm = in_norm;
            //v_text = in_text;
        }
    ''',
    fragment_shader='''
        #version 330

        uniform vec3 Light;
        uniform vec3 Color;
        //uniform bool UseTexture;
        //uniform sampler2D Texture;

        in vec3 v_vert;
        //in vec3 v_norm;
        //in vec2 v_text;

        out vec4 f_color;

        void main() {
            //float lum = clamp(dot(normalize(Light - v_vert), normalize(v_norm)), 0.0, 1.0) * 0.8 + 0.2;
            float lum = 1;
            //if (UseTexture) {
            //    f_color = vec4(texture(Texture, v_text).rgb * lum, 1.0);
            //} else {
                f_color = vec4(Color * lum, 1.0);
            //}
        }
    ''',
)

random_scaling = np.random.uniform(0, 1, 160)
random_scaling[random_scaling < 0.7] = 0.7
# random_scaling[random_scaling > 0.7] = 0.7

verts, faces = sphere_vertices(CUBE_FACE, 5)

print("basic vertices: ", sum([len(v.v) for v in verts]))
unique_verts = get_unique_vertices(verts)
print("unique vertices: ", len(unique_verts))

scale_vertices(faces, unique_verts, random_scaling)

vertex_buffers = []
for face in faces:
    verts = np.dstack([face.x(), face.y(), face.z()])
    vbo = ctx.buffer(verts.astype('f4').tobytes())
    vao = ctx.simple_vertex_array(prog, vbo, 'in_vert')
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

f, axarr = plt.subplots(1,7, sharex=True, sharey=True, figsize=(20, 5))

for rot in np.arange(0, 3.1, 0.5):
    fbo.clear(0.0, 0.0, 0.0, 0.0)
    rotate = Matrix44.from_z_rotation(rot)
    prog['Mvp'].write((proj * lookat * rotate).astype('f4').tobytes())
    # prog['Mvp'].write((proj * lookat).astype('f4').tobytes())
    # prog['Light'].value = (67.69, -8.14, 52.49)

    for vb, color in zip(vertex_buffers, FACE_COLORS):
        prog['Color'].value = color
        vb.render(moderngl.TRIANGLES)

    img = Image.frombytes('RGB', fbo.size, fbo.read(), 'raw', 'RGB', 0, -1)

    # plt.subplot(171 + (rot * 2))
    axarr[int(rot * 2)].imshow(img)

plt.tight_layout()
plt.show()
