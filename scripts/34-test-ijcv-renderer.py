from threedee_tools.datasets import CubeLoader
from threedee_tools.renderer import Renderer
import numpy as np
import matplotlib.pyplot as plt

env = Renderer(128, 128, shape="ijcv")

gen = CubeLoader()
imga = gen.sample()

print(gen.cam)
print(gen.light)

env.base_light = -gen.light+1
imgb = env.render(np.ones((160)), np.array([0, 0, 0]), cam_pos=gen.cam+.7)
imgb = np.array(imgb, dtype=np.float32) / 255

imgab = np.zeros((128, 128 * 2, 3), dtype=np.float32)
imgab[:, :128, :] = imga
imgab[:, 128:, :] = imgb

plt.imshow(imgab)
plt.show()
