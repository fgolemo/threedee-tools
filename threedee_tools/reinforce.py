import math
from torch import nn
import torch

pi = torch.FloatTensor([math.pi])
if torch.cuda.is_available():
    pi = pi.cuda()


def normal(x, mu, sigma_sq):
    a = (-1 * (x - mu).pow(2) / (2 * sigma_sq)).exp()
    b = 1 / (2 * sigma_sq * pi.expand_as(sigma_sq)).sqrt()
    return a * b


class ReinforcePolicy(nn.Module):
    def __init__(self, num_latents, num_outputs):
        super(ReinforcePolicy, self).__init__()

        self.relu = nn.ReLU()

        # inference part
        self.conv1 = nn.Conv2d(3, 32, (3, 3), (1, 1), (1, 1))
        self.conv2 = nn.Conv2d(32, 64, (3, 3), (2, 2), (1, 1))
        self.conv3 = nn.Conv2d(64, 128, (3, 3), (2, 2), (1, 1))

        # inference: conv to latent
        self.linear1_mu = nn.Linear(128 * 16 * 16, num_latents)
        self.linear1_stddev = nn.Linear(128 * 16 * 16, num_latents)

        self.linear2 = nn.Linear(num_latents, num_outputs)
        self.linear3 = nn.Linear(num_outputs, num_outputs)

    def encode(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))

        # mu, logprob
        return torch.sigmoid(self.linear1_mu(x.view(-1, 128 * 16 * 16))), torch.sigmoid(
            self.linear1_stddev(x.view(-1, 128 * 16 * 16)))

    def decode(self, z):
        x = self.relu(self.linear2(z))
        return torch.sigmoid(self.linear3(x))

    def forward(self, x):
        raise NotImplementedError("shouldn't use this directly")
