import itertools

import numpy as np


class Vec3D(object):
    def __init__(self, x, y, z):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def arr(self):
        return np.array((self.x, self.y, self.z), dtype=np.float32)

    def __eq__(self, other):
        # return self.x == other.x and self.y == other.y and self.z == other.z
        return np.linalg.norm(self.arr() - other.arr()) < 0.001  # because small imprecisions in point calc

    def __hash__(self):
        return hash((self.x, self.y, self.z))

    def scale(self, factor):
        self.x *= factor
        self.y *= factor
        self.z *= factor

    def __str__(self):
        return "({},{},{})".format(self.x, self.y, self.z)

    def copy(self):
        return Vec3D(self.x, self.y, self.z)


class Face(object):
    def __init__(self, origin, right, up):
        self.origin = origin
        self.up = up
        self.right = right


def grouper(iterable, n, padvalue=None):
  "grouper(3, 'abcdefg', 'x') --> ('a','b','c'), ('d','e','f'), ('g','x','x')"
  return itertools.zip_longest(*[iter(iterable)] * n, fillvalue=padvalue)

class Vertices(object):
    def __init__(self):
        self.v = []
        self.n = []
        # self.

    def copy(self):
        v = Vertices()
        v.v = self.v[:]
        return v

    def x(self):
        return [p.x for p in self.v]

    def y(self):
        return [p.y for p in self.v]

    def z(self):
        return [p.z for p in self.v]

    def nx(self):
        return [p.x for p in self.n]

    def ny(self):
        return [p.y for p in self.n]

    def nz(self):
        return [p.z for p in self.n]

    def calculate_normals(self):
        for a, b, c in grouper(self.v, 3):
            x = np.mean([a.x, b.x, c.x])
            y = np.mean([a.y, b.y, c.y])
            z = np.mean([a.z, b.z, c.z])
            for _ in range(3):
                self.n.append(Vec3D(x, y, z))

    def getQuad(self, a, b, c, d):
        out = []

        # triangle 1
        out.append(self.v[a].copy())
        out.append(self.v[b].copy())
        out.append(self.v[c].copy())

        # triangle 2
        out.append(self.v[a].copy())
        out.append(self.v[c].copy())
        out.append(self.v[d].copy())

        return out

    def getQuadAlt(self, a, b, c, d):
        out = []

        # triangle 1
        out.append(self.v[a].copy())
        out.append(self.v[b].copy())
        out.append(self.v[d].copy())

        # triangle 2
        out.append(self.v[b].copy())
        out.append(self.v[c].copy())
        out.append(self.v[d].copy())

        return out


CUBE_FACE = [
    Face(Vec3D(-1, -1, -1), Vec3D(2, 0, 0), Vec3D(0, 2, 0)),
    Face(Vec3D(1, -1, -1), Vec3D(0, 0, 2), Vec3D(0, 2, 0)),
    Face(Vec3D(1, -1, 1), Vec3D(-2, 0, 0), Vec3D(0, 2, 0)),

    Face(Vec3D(-1, -1, 1), Vec3D(0, 0, -2), Vec3D(0, 2, 0)),
    Face(Vec3D(-1, 1, -1), Vec3D(2, 0, 0), Vec3D(0, 0, 2)),
    Face(Vec3D(-1, -1, 1), Vec3D(2, 0, 0), Vec3D(0, 0, -2))
]

FACE_COLORS = [
    (0, 0, 1),
    (0, 1, 0),
    (1, 1, 0),
    (1, 0, 0),
    (0, 1, 1),
    (1, 1, 1),
]


def sphere_vertices(faces, subdiv_count):
    # This is mostly from https://github.com/caosdoar/spheres
    # Huge thanks!

    sphere_vertices = []
    sphere_face_vertices = []

    step = 1.0 / subdiv_count
    for f in faces:
        vertices = Vertices()
        origin = f.origin.arr()
        right = f.right.arr()
        up = f.up.arr()
        for j in range(subdiv_count + 1):
            for i in range(subdiv_count + 1):
                p = origin + step * (right * i + up * j)
                p2 = p * p
                rx = p[0] * np.sqrt(1.0 - 0.5 * (p2[1] + p2[2]) + p2[1] * p2[2] / 3.0)
                ry = p[1] * np.sqrt(1.0 - 0.5 * (p2[2] + p2[0]) + p2[2] * p2[0] / 3.0)
                rz = p[2] * np.sqrt(1.0 - 0.5 * (p2[0] + p2[1]) + p2[0] * p2[1] / 3.0)
                vertices.v.append(Vec3D(rx, ry, rz))
        sphere_vertices.append(vertices)

    k = subdiv_count + 1
    for face in range(len(faces)):
        vertices = Vertices()
        for j in range(subdiv_count):
            bottom = j < (subdiv_count / 2)

            for i in range(subdiv_count):
                left = i < (subdiv_count / 2)

                a = j * k + i
                b = j * k + i + 1
                c = (j + 1) * k + i
                d = (j + 1) * k + i + 1
                if (bottom and not left) or (left and not bottom):
                    verts = sphere_vertices[face].getQuadAlt(a, c, d, b)
                else:
                    verts = sphere_vertices[face].getQuad(a, c, d, b)

                vertices.v += verts

        sphere_face_vertices.append(vertices)

    return sphere_vertices, sphere_face_vertices


def get_unique_vertices(verts):
    all_vertices = []
    for v in verts:
        all_vertices += v.v
    all_vertices = list(set(all_vertices))
    return all_vertices


def get_scale_for_vert(verts, scales, search):
    for i in range(len(verts)):
        if verts[i] == search:
            return scales[i]


def scale_vertices(faces, verts, scales, iiis=None):
    cnt = 0
    for f in faces:
        for v in f.v:
            if iiis is None:  # very slow
                v.scale(get_scale_for_vert(verts, scales, v))
            else:  # better
                v.scale(scales[iiis[cnt]])
                cnt += 1


def get_scale_for_vert_dry(verts, search):
    for i in range(len(verts)):
        if verts[i] == search:
            return i


def scale_vertices_dry(faces, verts):
    iiis = []
    for f in faces:
        for v in f.v:
            iiis.append(get_scale_for_vert_dry(verts, v))

    return iiis
