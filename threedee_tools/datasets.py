from threedee_tools.renderer import Renderer
import numpy as np


class CubeGenerator(object):
    def __init__(self, width, height):
        self.renderer = Renderer(width, height, "cube", False)

    def sample(self):
        return self.renderer.render(np.zeros(160), np.random.uniform(-1, 1, 3))


class RandomSingleViewGenerator(object):
    def __init__(self, width, height, smin=0, smax=1):
        self.renderer = Renderer(width, height, "sphere", True)
        self.cam = np.random.uniform(-1, 1, 3)
        self.min = smin
        self.max = smax

    def sample(self):
        return self.renderer.render(np.random.uniform(self.min, self.max, 160), self.cam)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # cg = CubeGenerator(128,128)
    # while True:
    #     cube = cg.sample()
    #     plt.imshow(cube)
    #     plt.show()

    gen = RandomSingleViewGenerator(128, 128, .5, 1)
    while True:
        sample = gen.sample()
        plt.imshow(sample)
        plt.show()
