import gym
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class NormalizedActions(gym.ActionWrapper):

    def _action(self, action):
        action = np.clip(action, -1, 1)
        action = (action + 1) / 2  # [-1, 1] => [0, 1]
        action *= (self.action_space.high - self.action_space.low)
        action += self.action_space.low
        return action

    def _reverse_action(self, action):
        action -= self.action_space.low
        action /= (self.action_space.high - self.action_space.low)
        action = action * 2 - 1
        return action


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        """
        :param output1:
        :param output2:
        :param label: 0 - same image, 1 - different
        :return:
        """
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean(
            (1 - label) * torch.pow(euclidean_distance, 2) +
            (label) * torch.pow(
                torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive
