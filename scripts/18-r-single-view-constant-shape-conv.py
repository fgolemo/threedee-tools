import os
from time import strftime

import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import torch

from threedee_tools.datasets import CubeGenerator, RandomSingleViewGenerator, ConstantShapeGenerator
from threedee_tools.reinforce import REINFORCE
from threedee_tools.renderer import Renderer

GAMMA = 1  # discount factor - doesn't apply here
SEED = 123
NUM_STEPS = 100  # episode steps
NUM_EPISODES = 1000
HIDDEN_SIZE = 128
CHKP_FREQ = 100  # model saving freq
WIDTH = 64
HEIGHT = 64
REWARD_BUF = 1000

npa = np.array

from hyperdash import Experiment

exp_name = "3DR-18-constant-shape-conv"
exp_dir = exp_name + "-" + strftime("%Y%m%d%H%M%S")

exp = Experiment(exp_name)


class Policy(nn.Module):
    def __init__(self, hidden_size, num_inputs, num_outputs):
        super(Policy, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(3, 32, (3, 3), (1, 1), (1, 1))
        self.conv2 = nn.Conv2d(32, 64, (3, 3), (2, 2), (1, 1))
        self.linear3 = nn.Linear(64 * 32 * 32, num_outputs)
        self.linear3_ = nn.Linear(64 * 32 * 32, num_outputs)

    def forward(self, inputs):
        x = self.relu(self.conv1(inputs))
        x = self.relu(self.conv2(x))
        x = x.view(-1, 64 * 32 * 32)
        mu = torch.sigmoid(self.linear3(x))
        sigma_sq = torch.tanh(self.linear3_(x))

        return mu, sigma_sq


env = Renderer(WIDTH, HEIGHT)

data_generator = ConstantShapeGenerator(WIDTH, HEIGHT)

torch.manual_seed(SEED)
np.random.seed(SEED)

agent = REINFORCE(HIDDEN_SIZE, WIDTH * HEIGHT * 3, 160, Policy)

if not os.path.exists(exp_dir):
    os.mkdir(exp_dir)

reward_avg = []

for i_episode in range(NUM_EPISODES):
    target = data_generator.sample()
    env_state = np.ones(160, dtype=np.float32) * .5
    env_rot = np.zeros(3, dtype=np.float32)
    state = torch.Tensor([np.swapaxes(npa(target), 0, 2)])
    if torch.cuda.is_available():
        state = state.cuda()
    entropies = []
    log_probs = []
    rewards = []
    reward_raw_log = []  # just for logging purposes

    for t in range(NUM_STEPS):
        action, log_prob, entropy = agent.select_action(state)
        action = action.cpu()

        next_state = env.render(action[0], data_generator.cam)

        reward_raw = -np.linalg.norm(npa(target) - npa(next_state)).sum()
        reward_raw_log.append(reward_raw)

        if len(reward_avg) == 0:
            reward = reward_raw
        else:
            reward = reward_raw - np.mean(reward_avg)

        # update running mean
        reward_avg.append(reward_raw)
        if len(reward_avg) > REWARD_BUF:
            reward_avg.pop(0)

        rewards.append(reward)
        entropies.append(entropy[0])
        log_probs.append(log_prob[0])

    agent.update_parameters(rewards, log_probs, entropies, GAMMA)

    if i_episode % CHKP_FREQ == 0:
        torch.save(agent.model.state_dict(), os.path.join(exp_dir, 'reinforce-' + str(i_episode) + '.pkl'))

    # print("Episode: {}, reward: {}".format(i_episode, np.sum(rewards)))
    exp.metric("episode", i_episode)
    exp.metric("rewards", np.mean(reward_raw_log))

    del rewards
    del log_probs
    del entropies
    del state
