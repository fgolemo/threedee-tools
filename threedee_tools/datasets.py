import os

import torch
from threediqtt.dataset import ValDataset, TestDataset

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


class RotatingCubeGenerator(object):
    def __init__(self, width, height):
        self.renderer = Renderer(width, height, "cube", False)
        self.cam = None

    def sample(self):
        self.cam = np.random.uniform(-1, 1, 3)
        return self.renderer.render(np.zeros(160), self.cam)


class RandomSingleViewGenerator(object):
    def __init__(self, width, height, smin=0, smax=1):
        self.renderer = Renderer(width, height, "sphere", True)
        self.cam = np.random.uniform(-1, 1, 3)
        self.min = smin
        self.max = smax

    def sample(self):
        return self.renderer.render(np.random.uniform(self.min, self.max, 160), self.cam)


class RotatingRandomShapeGenerator(object):
    def __init__(self, width, height, smin=.4, smax=.8):
        self.renderer = Renderer(width, height, "sphere", True)
        self.shape = np.random.uniform(smin, smax, 160)
        self.cam = None

    def sample(self):
        self.cam = np.random.uniform(-1, 1, 3)
        return self.renderer.render(self.shape, self.cam)


class ConstantShapeGenerator(object):
    def __init__(self, width, height):
        self.renderer = Renderer(width, height, "sphere", True)
        self.cam = np.random.uniform(-1, 1, 3)

    def sample(self):
        shape = np.ones(160) * .5
        return self.renderer.render(shape, self.cam)


class RotatingConstantShapeGenerator(object):
    def __init__(self, width, height, radius=.5):
        self.renderer = Renderer(width, height, "sphere", True)
        self.shape = np.ones(160) * radius

    def sample(self):
        self.cam = np.random.uniform(-1, 1, 3)
        return self.renderer.render(self.shape, self.cam)


class CubeSphereComparisonGenerator(object):

    def __init__(self, width, height):
        self.cube = Renderer(width, height, "cube", False)
        self.sphere = Renderer(width, height, "sphere", True)
        self.shape = np.ones(160) * 0.9

    def sample(self):
        cam = np.random.uniform(-1, 1, 3)
        return self.cube.render(np.zeros(160), cam), self.sphere.render(self.shape, cam)


class RotatingSingle3DIQTTGenerator(object):
    def __init__(self, width, height, smin=.5, smax=1):
        self.renderer = Renderer(width, height, "iqtt", True)
        self.shape = np.random.uniform(smin, smax, 160)

    def sample(self, cam=None):
        if cam is None:
            self.cam = np.random.uniform(-1, 1, 3)
        else:
            self.cam = cam
        return self.renderer.render(self.shape, self.cam)


class IQTTLoader(object):
    def __init__(self, greyscale=False):
        self.ds = TestDataset(os.path.expanduser("~/Downloads/3diqtt-v2-test.h5"))
        self.gs = greyscale

    def sample(self, as_np=False):
        random_idx = np.random.randint(0, len(self.ds))
        img_q = self.ds[random_idx]["question"][0]

        if self.gs:
            img_q = torch.sum(img_q, dim=0, keepdim=True) / 3
            img_q = torch.stack((img_q[0, :, :], img_q[0, :, :], img_q[0, :, :]))

        if as_np:
            return img_q.permute(1, 2, 0).numpy()
        else:
            return img_q


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    #### Different basic shape gens

    # gen = CubeSingleViewGenerator(128,128)
    # gen = RandomSingleViewGenerator(128, 128, .5, 1)
    # gen = RotatingConstantShapeGenerator(128, 128, .7)
    # gen = RotatingRandomShapeGenerator(128, 128)
    # gen = RotatingSingle3DIQTTGenerator(128, 128)
    gen = IQTTLoader(greyscale=True)
    while True:
        sample = gen.sample(as_np=True)
        # print (np.min(sample), np.max(sample))
        plt.imshow(sample)
        plt.show()

    #### ROTATING REZENDE

    # gen = RotatingSingle3DIQTTGenerator(128, 128)
    # image = np.random.uniform(0, 254, size=(128, 128, 3))
    # fig, ax = plt.subplots()
    # image_container = ax.imshow(image)
    # rot_x = -1
    # while True:
    #     sample = gen.sample(cam=np.array([0, rot_x, rot_x]))
    #     image_container.set_data(sample)
    #     # greyscale = np.sum(sample, axis=2)/(255*3)
    #     # greyscale = np.dstack((greyscale, greyscale, greyscale))
    #     # image_container.set_data(greyscale)
    #     fig.canvas.draw()
    #     plt.pause(0.01)
    #     rot_x += 0.01
    #     if rot_x > 1:
    #         rot_x = -1

    # JUST MAKING SURE THE CUBE COLORS ARE COHERENT WITH REZENDE

    # gen = CubeSphereComparisonGenerator(128, 128)
    # while True:
    #     cube, sphere = gen.sample()
    #     plt.subplot(2, 1, 1)
    #     plt.imshow(cube)
    #     plt.subplot(2, 1, 2)
    #     plt.imshow(sphere)
    #     plt.tight_layout()
    #     plt.show()
