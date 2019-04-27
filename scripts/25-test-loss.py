import torch
import numpy as np
import torch.nn.functional as F

size = (64, 64, 3)

b = np.zeros(size, dtype=np.float32)
bt = torch.tensor(b).float()

losses = []
for i in range(100):
    a = np.random.uniform(0, 1, size)
    at = torch.tensor(a).float()

    loss = -F.binary_cross_entropy(at, bt, reduction="sum")
    losses.append(loss.item())

losses = losses - np.mean(losses)

print(np.sum(losses))

print("rand", torch.randn(10))

for i in range(8):
    print("norm"+str(i), torch.normal(mean=torch.zeros(10), std=.2 + (i / 10)))
