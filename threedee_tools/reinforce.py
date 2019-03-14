import math

import torch.optim as optim
import torch.nn.utils as utils
import torch.nn.functional as F
import torch

pi = torch.FloatTensor([math.pi])
if torch.cuda.is_available():
    pi = pi.cuda()

def normal(x, mu, sigma_sq):
    a = (-1 * (x - mu).pow(2) / (2 * sigma_sq)).exp()
    b = 1 / (2 * sigma_sq * pi.expand_as(sigma_sq)).sqrt()
    return a * b


class REINFORCE:
    def __init__(self, hidden_size, num_inputs, action_space, policy, steps=100):
        self.action_space = action_space
        self.steps = steps
        self.model = policy(hidden_size, num_inputs, action_space)
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.model.train()

    def select_action(self, state):
        # mu, sigma_sq = self.model(state.cuda())
        mu, sigma_sq = self.model(state)
        sigma_sq = F.softplus(sigma_sq)

        # eps = torch.randn(mu.size()).cuda()
        eps = torch.randn(mu.size())
        if torch.cuda.is_available():
            eps = eps.cuda()
        # calculate the probability
        action = (mu + sigma_sq.sqrt() * eps).data
        prob = normal(action, mu, sigma_sq)
        entropy = -0.5 * ((sigma_sq + 2 * pi.expand_as(sigma_sq)).log() + 1)

        log_prob = prob.log()
        return action, log_prob, entropy

    def update_parameters(self, rewards, log_probs, entropies, gamma):
        R = torch.zeros(self.action_space)
        loss = 0
        for i in reversed(range(len(rewards))):
            R = gamma * R + rewards[i]
            # loss = loss - (log_probs[i] * (R.expand_as(log_probs[i])).cuda()).sum() - (
            #         0.0001 * entropies[i].cuda()).sum()
            R_exp = R.expand_as(log_probs[i])
            if torch.cuda.is_available():
                R_exp = R_exp.cuda()
            loss = loss - (log_probs[i] * R_exp).sum() - (0.0001 * entropies[i]).sum()
        loss = loss / len(rewards)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 40)
        self.optimizer.step()
        del loss

