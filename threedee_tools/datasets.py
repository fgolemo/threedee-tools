from threedee_tools.renderer import Renderer
import numpy as np


class CubeGenerator(object):
    def __init__(self, width, height):
        self.renderer = Renderer(width, height, "cube", False)

    def sample(self):
        return self.renderer.render(np.zeros(160), np.random.uniform(-1, 1, 3))


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    cg = CubeGenerator(128,128)
    while True:
        cube = cg.sample()
        plt.imshow(cube)
        plt.show()