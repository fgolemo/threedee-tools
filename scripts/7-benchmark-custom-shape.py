import copy
import time

import moderngl
import numpy as np
from threedee_tools.utils import sphere_vertices, CUBE_FACE, FACE_COLORS, get_unique_vertices, scale_vertices, \
    scale_vertices_dry
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

verts_base, faces_base = sphere_vertices(CUBE_FACE, 5)
unique_verts = get_unique_vertices(verts_base)
iiis = scale_vertices_dry(faces_base, unique_verts)

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
prog['Mvp'].write((proj * lookat).astype('f4').tobytes())


runs = 100
timer_copy = []
timer_scaling = []
timer_buffering = []
timer_rendering = []
timer_img = []
# avg time / frame: 0.7709s, Hz: 1.3

start_total = time.time()
for i in range(runs):
    start = time.time()
    # faces = copy.deepcopy(faces_base)
    faces = [f.copy() for f in faces_base]
    timer_copy.append(time.time()-start)

    start = time.time()
    random_scaling = np.random.uniform(0, 1, 160)
    scale_vertices(faces, unique_verts, random_scaling, iiis)
    timer_scaling.append(time.time() - start)

    start = time.time()
    vertex_buffers = []
    for face in faces:
        verts = np.dstack([face.x(), face.y(), face.z()])
        vbo = ctx.buffer(verts.astype('f4').tobytes())
        vao = ctx.simple_vertex_array(prog, vbo, 'in_vert')
        vertex_buffers.append(vao)
    timer_buffering.append(time.time() - start)

    start = time.time()
    fbo.clear(0.0, 0.0, 0.0, 0.0)
    for vb, color in zip(vertex_buffers, FACE_COLORS):
        prog['Color'].value = color
        vb.render(moderngl.TRIANGLES)
    timer_rendering.append(time.time() - start)

    start = time.time()
    img = Image.frombytes('RGB', fbo.size, fbo.read(), 'raw', 'RGB', 0, -1)
    timer_img.append(time.time() - start)

diff = time.time()-start_total
print ("avg time / frame: {}s, Hz: {} | total".format(np.around(diff/runs,4), np.around(runs/diff,2)))

for name, timer in [
    ("copying",timer_copy),
    ("scaling",timer_scaling),
    ("buffering",timer_buffering),
    ("rendering",timer_rendering),
    ("img",timer_img),
]:
    print("avg time / frame: {}s, Hz: {} | {}".format(
        np.around(np.mean(timer), 4),
        np.around(1 / np.mean(timer), 2),
        name
    ))
