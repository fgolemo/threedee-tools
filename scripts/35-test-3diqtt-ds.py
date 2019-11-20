from threedee_tools.datasets import IQTTLoader
import matplotlib.pyplot as plt
import numpy as np

data_generator = IQTTLoader(greyscale=True)
print(len(data_generator.ds))

for i in range(3):
    img_q, img_a, img_d1, img_d2 = data_generator.sample_qa(as_np=True)

    print(np.min(img_a), np.max(img_a), np.mean(img_a))

    fig, axes = plt.subplots(2, 2)
    axes[0, 0].imshow(img_q)
    axes[0, 0].set_title("Question")
    axes[0, 1].imshow(img_a)
    axes[0, 1].set_title("Answer")
    axes[1, 0].imshow(img_d1)
    axes[1, 0].set_title("Distractor 1")
    axes[1, 1].imshow(img_d2)
    axes[1, 1].set_title("Distractor 2")
    plt.show()
