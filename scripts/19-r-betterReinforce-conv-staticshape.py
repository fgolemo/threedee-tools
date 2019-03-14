import os
from time import strftime

import numpy as np

import torch.nn as nn
import torch
from torch.distributions import Normal

from threedee_tools.datasets import CubeGenerator, RandomSingleViewGenerator, ConstantShapeGenerator
from threedee_tools.renderer import Renderer

SEED = 123
SAMPLES = 100
NUM_EPISODES = 1000
LATENT_SIZE = 10
CHKP_FREQ = 100  # model saving freq
WIDTH = 64
HEIGHT = 64

npa = np.array

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from hyperdash import Experiment

exp_name = "3DR-19-newReinforce-constant-shape-conv"
exp_dir = exp_name + "-" + strftime("%Y%m%d%H%M%S")

exp = Experiment(exp_name)


class Policy(nn.Module):
    def __init__(self, num_latents, num_outputs):
        super(Policy, self).__init__()

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

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
        return self.sigmoid(self.linear1_mu(x.view(-1, 128 * 16 * 16))), self.sigmoid(
            self.linear1_stddev(x.view(-1, 128 * 16 * 16)))

    def decode(self, z):
        x = self.relu(self.linear2(z))
        return self.sigmoid(self.linear3(x))

    def forward(self, x):
        # mu, logvar = self.encode(x)
        # z = self.reparameterize(mu, logvar)
        # return self.decode(z), mu, logvar
        raise NotImplementedError("shouldn't use this directly")


env = Renderer(WIDTH, HEIGHT)

data_generator = ConstantShapeGenerator(WIDTH, HEIGHT)

torch.manual_seed(SEED)
np.random.seed(SEED)

policy = Policy(LATENT_SIZE, 160).to(device)

optimizer = torch.optim.Adam(policy.parameters())
eps = np.finfo(np.float32).eps.item()

if not os.path.exists(exp_dir):
    os.mkdir(exp_dir)

import matplotlib.pyplot as plt

for i_episode in range(NUM_EPISODES):
    # sample image
    state_raw = data_generator.sample()

    state = torch.Tensor([np.swapaxes(npa(state_raw), 0, 2)]).to(device)

    # encode to latent variables (mu/var)
    latent_mu, latent_stddev = policy.encode(state)

    rewards = []
    rewards_raw = []
    log_probs = []

    for k in range(SAMPLES):

        # sample K times
        m = Normal(latent_mu, latent_stddev)
        action = m.rsample()
        log_probs.append(m.log_prob(action))

        params = policy.decode(action)

        # render out an image for each of the K samples
        # IMPORTANT THIS CURRENTLY ASSUMES BATCH SIZE = 1
        next_state = env.render(params.detach().view(-1).numpy(), data_generator.cam)

        # calculate reward for each one of the K samples
        reward_raw = -(np.square(npa(state_raw) - npa(next_state))).mean(axis=None)
        rewards_raw.append(reward_raw)

    # deduct average reward of all K-1 samples (variance reduction)
    for k in range(SAMPLES):
        baseline = np.mean(rewards_raw[:k] + rewards_raw[k + 1:])
        rewards.append(rewards_raw[k] - baseline)


    # calculate additional VAE loss
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + torch.log(latent_stddev.pow(2)) - latent_mu.pow(2) - latent_stddev.pow(2))

    # update model
    returns = torch.Tensor(rewards).to(device)
    # returns = (returns - returns.mean()) / (returns.std() + eps)

    policy_loss = []
    for log_prob, R in zip(log_probs, returns):
        policy_loss.append(-log_prob * R)

    optimizer.zero_grad()
    policy_loss_sum = torch.cat(policy_loss).sum() + KLD
    loss_copy = policy_loss_sum.detach().numpy().copy()
    policy_loss_sum.backward()
    optimizer.step()

    if i_episode % CHKP_FREQ == 0:
        torch.save(policy.state_dict(), os.path.join(exp_dir, 'reinforce-' + str(i_episode) + '.pkl'))

    exp.metric("episode", i_episode)
    exp.metric("rewards", np.mean(rewards_raw))
    exp.metric("loss", float(loss_copy))
