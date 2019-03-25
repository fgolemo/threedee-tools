from threedee_tools.renderer import Renderer
import numpy as np


class CubeGenerator(object):
    def __init__(self, width, height):
        self.renderer = Renderer(width, height, "cube", False)

    def sample(self):
        return self.renderer.render(np.zeros(160), np.random.uniform(-1, 1, 3))


class CubeSingleViewGenerator(object):
    def __init__(self, width, height):
        self.renderer = Renderer(width, height, "cube", False)
        self.cam = np.random.uniform(-1, 1, 3)

    def sample(self):
        return self.renderer.render(np.zeros(160), self.cam)


class RandomSingleViewGenerator(object):
    def __init__(self, width, height, smin=0, smax=1):
        self.renderer = Renderer(width, height, "sphere", True)
        self.cam = np.random.uniform(-1, 1, 3)
        self.min = smin
        self.max = smax

    def sample(self):
        return self.renderer.render(np.random.uniform(self.min, self.max, 160), self.cam)


class ConstantShapeGenerator(object):
    def __init__(self, width, height):
        self.renderer = Renderer(width, height, "sphere", True)
        self.cam = np.random.uniform(-1, 1, 3)

    def sample(self):
        shape = np.ones(160) * .5
        return self.renderer.render(shape, self.cam)


class CubeSphereComparisonGenerator(object):

    def __init__(self, width, height):
        self.cube = Renderer(width, height, "cube", False)
        self.sphere = Renderer(width, height, "sphere", True)
        self.shape = np.ones(160) * 0.9

    def sample(self):
        cam = np.random.uniform(-1, 1, 3)
        return self.cube.render(np.zeros(160), cam), self.sphere.render(self.shape, cam)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # cg = CubeGenerator(128,128)
    # while True:
    #     cube = cg.sample()
    #     plt.imshow(cube)
    #     plt.show()

    # gen = RandomSingleViewGenerator(128, 128, .5, 1)
    # while True:
    #     sample = gen.sample()
    #     plt.imshow(sample)
    #     plt.show()

    gen = CubeSphereComparisonGenerator(128, 128)
    while True:
        cube, sphere = gen.sample()
        plt.subplot(2, 1, 1)
        plt.imshow(cube)
        plt.subplot(2, 1, 2)
        plt.imshow(sphere)
        plt.tight_layout()
        plt.show()
