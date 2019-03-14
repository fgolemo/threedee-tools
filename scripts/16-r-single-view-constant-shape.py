import os
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

exp = Experiment("3DR-16-constant-shape")


class Policy(nn.Module):
    def __init__(self, hidden_size, num_inputs, num_outputs):
        super(Policy, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, num_outputs)
        self.linear3_ = nn.Linear(hidden_size, num_outputs)

    def forward(self, inputs):
        x = inputs
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        mu = torch.sigmoid(self.linear3(x))
        sigma_sq = torch.tanh(self.linear3_(x))

        return mu, sigma_sq


env = Renderer(WIDTH, HEIGHT)

data_generator = ConstantShapeGenerator(WIDTH, HEIGHT)

torch.manual_seed(SEED)
np.random.seed(SEED)

agent = REINFORCE(HIDDEN_SIZE, WIDTH * HEIGHT * 3, 160, Policy)

dir = 'ckpt_3dreinforcev3'
if not os.path.exists(dir):
    os.mkdir(dir)

reward_avg = []

for i_episode in range(NUM_EPISODES):
    target = data_generator.sample()
    env_state = np.ones(160, dtype=np.float32) * .5
    env_rot = np.zeros(3, dtype=np.float32)
    state = torch.Tensor([npa(target)]).view(-1)
    if torch.cuda.is_available():
        state = state.cuda()
    entropies = []
    log_probs = []
    rewards = []
    reward_raw_log = [] # just for logging purposes

    for t in range(NUM_STEPS):
        action, log_prob, entropy = agent.select_action(state)
        action = action.cpu()

        next_state = env.render(action, data_generator.cam)

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
        entropies.append(entropy)
        log_probs.append(log_prob)

    agent.update_parameters(rewards, log_probs, entropies, GAMMA)

    if i_episode % CHKP_FREQ == 0:
        torch.save(agent.model.state_dict(), os.path.join(dir, 'reinforce-' + str(i_episode) + '.pkl'))

    # print("Episode: {}, reward: {}".format(i_episode, np.sum(rewards)))
    exp.metric("episode", i_episode)
    exp.metric("rewards", np.mean(reward_raw_log))

    del rewards
    del log_probs
    del entropies
    del state