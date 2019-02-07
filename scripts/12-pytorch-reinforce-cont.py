import math
import os
import numpy as np

import gym
from gym import wrappers

import torch.nn as nn
import torch.nn.functional as F
import torch

from threedee_tools.reinforce import REINFORCE
from threedee_tools.utils_rl import NormalizedActions

ENV_NAME = "MountainCarContinuous-v0"
GAMMA = 0.99 # discount factor
SEED = 123
NUM_STEPS = 999 # episode steps
NUM_EPISODES = 2000
HIDDEN_SIZE = 128
RENDER = False
DEBUG = False
CHKP_FREQ = 100 # model saving freq

class Policy(nn.Module):
    def __init__(self, hidden_size, num_inputs, action_space):
        super(Policy, self).__init__()
        self.action_space = action_space
        num_outputs = action_space.shape[0]

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, num_outputs)
        self.linear2_ = nn.Linear(hidden_size, num_outputs)

    def forward(self, inputs):
        x = inputs
        x = F.relu(self.linear1(x))
        mu = self.linear2(x)
        sigma_sq = self.linear2_(x)

        return mu, sigma_sq

env = NormalizedActions(gym.make(ENV_NAME))

if DEBUG:
    env = wrappers.Monitor(env, '/tmp/{}-experiment'.format(ENV_NAME), force=True)

env.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

agent = REINFORCE(HIDDEN_SIZE, env.observation_space.shape[0], env.action_space, Policy)

dir = 'ckpt_' + ENV_NAME
if not os.path.exists(dir):
    os.mkdir(dir)

for i_episode in range(NUM_EPISODES):
    state = torch.Tensor([env.reset()])
    entropies = []
    log_probs = []
    rewards = []
    for t in range(NUM_STEPS):
        action, log_prob, entropy = agent.select_action(state)
        action = action.cpu()

        next_state, reward, done, _ = env.step(action.numpy()[0])

        entropies.append(entropy)
        log_probs.append(log_prob)
        rewards.append(reward)
        state = torch.Tensor([next_state])

        if RENDER:
            env.render()

        if done:
            break

    agent.update_parameters(rewards, log_probs, entropies, GAMMA)

    if i_episode % CHKP_FREQ == 0:
        torch.save(agent.model.state_dict(), os.path.join(dir, 'reinforce-' + str(i_episode) + '.pkl'))

    print("Episode: {}, reward: {}".format(i_episode, np.sum(rewards)))

env.close()