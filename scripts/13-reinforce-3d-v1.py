import os
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import torch

from threedee_tools.datasets import CubeGenerator
from threedee_tools.reinforce import REINFORCE
from threedee_tools.renderer import Renderer

GAMMA = 0.99  # discount factor
SEED = 123
NUM_STEPS = 100  # episode steps
NUM_EPISODES = 1000
HIDDEN_SIZE = 128
RENDER = False
CHKP_FREQ = 100  # model saving freq
WIDTH = 64
HEIGHT = 64
ALPHA_STATE = 1 / NUM_STEPS * 2
ALPHA_ROT = 1 / NUM_STEPS * 2

npa = np.array


class Policy(nn.Module):
    def __init__(self, hidden_size, num_inputs, num_outputs):
        super(Policy, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, num_outputs)
        self.linear2_ = nn.Linear(hidden_size, num_outputs)

    def forward(self, inputs):
        x = inputs
        x = F.relu(self.linear1(x))
        mu = torch.sigmoid(self.linear2(x))
        sigma_sq = torch.tanh(self.linear2_(x))

        return mu, sigma_sq


env = Renderer(WIDTH, HEIGHT)

cube_generator = CubeGenerator(WIDTH, HEIGHT)

torch.manual_seed(SEED)
np.random.seed(SEED)

agent = REINFORCE(HIDDEN_SIZE, WIDTH * HEIGHT * 3, 163, Policy)

dir = 'ckpt_3dreinforcev1'
if not os.path.exists(dir):
    os.mkdir(dir)

for i_episode in range(NUM_EPISODES):
    target = cube_generator.sample()
    env_state = np.ones(160, dtype=np.float32) * .5
    env_rot = np.zeros(3, dtype=np.float32)
    state = torch.Tensor([npa(target) - npa(env.render(env_state, env_rot))]).view(-1)
    entropies = []
    log_probs = []
    rewards = []
    for t in range(NUM_STEPS):
        action, log_prob, entropy = agent.select_action(state)
        action = action.cpu()

        env_state = env_state + ALPHA_STATE * action[:160]
        env_rot = env_rot + ALPHA_ROT * action[160:] * 2 - 1  # convert from [0,1] to [-1,1]

        next_state = env.render(env_state, env_rot)

        reward = -np.linalg.norm(npa(target) - npa(next_state)).sum()
        rewards.append(reward)

        entropies.append(entropy)
        log_probs.append(log_prob)

    agent.update_parameters(rewards, log_probs, entropies, GAMMA)

    if i_episode % CHKP_FREQ == 0:
        torch.save(agent.model.state_dict(), os.path.join(dir, 'reinforce-' + str(i_episode) + '.pkl'))

    print("Episode: {}, reward: {}".format(i_episode, np.sum(rewards)))
