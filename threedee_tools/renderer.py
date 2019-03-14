import moderngl
from PIL import Image
from pyrr import Matrix44
import numpy as np
from threedee_tools.shaders import VERTEX_SHADER_NORMAL, FARGMENT_SHADER_LIGHT_COLOR
from threedee_tools.utils_3d import sphere_vertices, CUBE_FACE, get_unique_vertices, \
    scale_vertices_dry, scale_vertices, FACE_COLORS, cube_vertices


class Renderer(object):

    def __init__(self, width, height, shape="sphere", scaled=True, sphere_subdiv=5):
        # this will be the main thing that the neural network changes
        self.vertex_values = np.ones(160, dtype=np.float32)  # in range [0,1]
        self.rotation_values = np.zeros(3, dtype=np.float32)  # in range [0,1] corresponding to 0 deg to 360 deg

        self.shape = shape
        self.scaled = scaled

        self.ctx = moderngl.create_standalone_context()
        self.prog = self.ctx.program(
            vertex_shader=VERTEX_SHADER_NORMAL,
            fragment_shader=FARGMENT_SHADER_LIGHT_COLOR)

        if shape == "sphere":
            self.verts_base, self.faces_base = sphere_vertices(CUBE_FACE, sphere_subdiv)
        elif shape == "cube":
            self.verts_base, self.faces_base = cube_vertices()
        else:
            assert NotImplementedError("Shape not implemented: '{}'".format(shape))

        self.unique_verts = get_unique_vertices(self.verts_base)
        self.iiis = scale_vertices_dry(self.faces_base, self.unique_verts)

        self.ctx.enable(moderngl.DEPTH_TEST)
        self.fbo = self.ctx.simple_framebuffer((width, height))
        self.fbo.use()

        self.proj = Matrix44.perspective_projection(45.0, width / height, 0.1, 1000.0)

        # light position (will stay in place, independent of object/cam rotation)
        self.base_light = np.array((10, 10, 10))

    def sanitize(self, vertex_values, rotation_values):
        return np.clip(vertex_values, 0, 1), \
               rotation_values * 2 * np.pi

    def render(self, vertex_values, rotation_values):
        vertex_values, rotation_values = self.sanitize(vertex_values, rotation_values)

        faces = [f.copy() for f in self.faces_base]

        if self.scaled:
            scale_vertices(faces, vertex_values, iiis=self.iiis)

        vertex_buffers = []
        for face in faces:
            face.calculate_normals()
            verts = np.dstack([face.x(), face.y(), face.z(), face.nx(), face.ny(), face.nz()])
            vbo = self.ctx.buffer(verts.astype('f4').tobytes())
            vao = self.ctx.simple_vertex_array(self.prog, vbo, 'in_vert', 'in_norm')
            vertex_buffers.append(vao)

        self.fbo.clear(0.0, 0.0, 0.0, 0.0)

        lookat = Matrix44.look_at(
            (2, 2, 1),  # eye / camera position
            (0.0, 0.0, 0.0),  # lookat
            (0.0, 0.0, 1.0),  # camera up vector
        )

        rotate = np.array(Matrix44.from_eulers(rotation_values))
        self.prog['Lights'].value = tuple(np.matmul(rotate[:3, :3], self.base_light).reshape(1, -1)[0])
        self.prog['Mvp'].write((self.proj * lookat * rotate).astype('f4').tobytes())

        for vb, color in zip(vertex_buffers, FACE_COLORS):
            self.prog['Color'].value = color
            vb.render(moderngl.TRIANGLES)

        return Image.frombytes('RGB', self.fbo.size, self.fbo.read(), 'raw', 'RGB', 0, -1)
