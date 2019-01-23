from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
import numpy as np


# namespace CubeToSphere
# {
# 	static const Vector3 origins[6] =
# 	{
# 		Vector3(-1.0, -1.0, -1.0),
# 		Vector3(1.0, -1.0, -1.0),
# 		Vector3(1.0, -1.0, 1.0),

# 		Vector3(-1.0, -1.0, 1.0),
# 		Vector3(-1.0, 1.0, -1.0),
# 		Vector3(-1.0, -1.0, 1.0)
# 	};
# 	static const Vector3 rights[6] =
# 	{
# 		Vector3(2.0, 0.0, 0.0),
# 		Vector3(0.0, 0.0, 2.0),
# 		Vector3(-2.0, 0.0, 0.0),

# 		Vector3(0.0, 0.0, -2.0),
# 		Vector3(2.0, 0.0, 0.0),
# 		Vector3(2.0, 0.0, 0.0)
# 	};
# 	static const Vector3 ups[6] =
# 	{
# 		Vector3(0.0, 2.0, 0.0),
# 		Vector3(0.0, 2.0, 0.0),
# 		Vector3(0.0, 2.0, 0.0),

# 		Vector3(0.0, 2.0, 0.0),
# 		Vector3(0.0, 0.0, 2.0),
# 		Vector3(0.0, 0.0, -2.0)
# 	};
# };


class Vec3D(object):
    def __init__(self, x, y, z):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def arr(self):
        return np.array((self.x, self.y, self.z), dtype=np.float32)


class Face(object):
    def __init__(self, origin, right, up):
        self.origin = origin
        self.up = up
        self.right = right


class Vertices(object):

    def __init__(self):
        self.v = []

    def x(self):
        return [p.x for p in self.v]

    def y(self):
        return [p.y for p in self.v]

    def z(self):
        return [p.z for p in self.v]


CUBE_FACE = [
    Face(Vec3D(-1, -1, -1), Vec3D(2, 0, 0), Vec3D(0, 2, 0)),
    Face(Vec3D(1, -1, -1), Vec3D(0, 0, 2), Vec3D(0, 2, 0)),
    Face(Vec3D(1, -1, 1), Vec3D(-2, 0, 0), Vec3D(0, 2, 0)),

    Face(Vec3D(-1, -1, 1), Vec3D(0, 0, -2), Vec3D(0, 2, 0)),
    Face(Vec3D(-1, 1, -1), Vec3D(2, 0, 0), Vec3D(0, 0, 2)),
    Face(Vec3D(-1, -1, 1), Vec3D(2, 0, 0), Vec3D(0, 0, -2))
]


def sphere_vertices(faces, subdiv_count):
    sphere_faces = []

    step = 1.0 / subdiv_count
    for f in faces:
        vertices = Vertices()
        origin = f.origin.arr()
        right = f.right.arr()
        up = f.up.arr()
        for j in range(subdiv_count+1):
            for i in range(subdiv_count+1):
                p = origin + step * (right * i + up * j)
                p2 = p * p
                rx = p[0]*np.sqrt(1.0 - 0.5 * (p2[1] + p2[2]) + p2[1] * p2[2] / 3.0)
                ry = p[1]*np.sqrt(1.0 - 0.5 * (p2[2] + p2[0]) + p2[2] * p2[0] / 3.0)
                rz = p[2]*np.sqrt(1.0 - 0.5 * (p2[0] + p2[1]) + p2[0] * p2[1] / 3.0)
                vertices.v.append(Vec3D(rx, ry, rz))
        sphere_faces.append(vertices)
    return sphere_faces


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

sphere_faces = sphere_vertices(CUBE_FACE, 6)

# for c, m, zlow, zhigh in [('r', 'o', -50, -25), ('b', '^', -30, -5)]:
# ax.scatter(xs, ys, zs, c=c, marker=m)

for vertices in sphere_faces:
    ax.scatter(vertices.x(), vertices.y(), vertices.z(), alpha=0.5)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()
