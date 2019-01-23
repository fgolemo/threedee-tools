import moderngl
import numpy as np

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

x = np.linspace(-1.0, 1.0, 50)
y = np.random.rand(50) - 0.5
z = np.zeros(50)
r = np.ones(50)
g = np.zeros(50)
b = np.zeros(50)

vertices = np.dstack([x, y, z])
print(vertices.shape)

vbo = ctx.buffer(vertices.astype('f4').tobytes())
vao = ctx.simple_vertex_array(prog, vbo, 'in_vert')

width = 512
height = 512

ctx.enable(moderngl.DEPTH_TEST)
fbo = ctx.simple_framebuffer((width, height))
fbo.use()
fbo.clear(1.0, 1.0, 1.0, 1.0)

proj = Matrix44.perspective_projection(45.0, width / height, 0.1, 1000.0)
lookat = Matrix44.look_at(
    (0, 1, 3),  # eye / camera position
    (0.0, 0.0, 0.0),  # lookat
    (0.0, 0.0, 1.0),  # camera up vector
)

rotate = Matrix44.from_z_rotation(np.sin(0) * 0.5 + 0.2)

prog['Mvp'].write((proj * lookat * rotate).astype('f4').tobytes())
# prog['Light'].value = (67.69, -8.14, 52.49)
prog['Color'].value = (1, 0, 0)
vao.render(moderngl.LINE_STRIP)

Image.frombytes('RGB', fbo.size, fbo.read(), 'raw', 'RGB', 0, -1).show()
