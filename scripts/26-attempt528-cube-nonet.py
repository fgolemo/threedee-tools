from comet_ml import Experiment
import os
from time import strftime

import numpy as np

import torch.nn as nn
import torch
from torch.nn import functional as F
from tqdm import trange

from threedee_tools.datasets import RotatingConstantShapeGenerator, RotatingRandomShapeGenerator, RotatingCubeGenerator
from threedee_tools.renderer import Renderer


params = {
    "SEED": 123,
    "SAMPLES": 100,
    "NUM_EPISODES": 1000,
    "LATENT_SIZE": 160,
    "CHKP_FREQ": 2,  # model saving freq
    "WIDTH": 64,
    "HEIGHT": 64,
    "LR": 1e-4,  # learning rate
    "KLD_WEIGHT": .1,  # learning rate
    "KLD_DELAY": 0,  # how many episodes do we skip adding the KLD to the loss
    "VARIANCE_WEIGHT": 2
}


npa = np.array

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pi = torch.Tensor([np.pi]).float().to(device)

exp_name = "24-newReparam"
exp_dir = "experiments/" + exp_name + "-" + strftime("%Y%m%d%H%M%S")

experiment = Experiment(api_key="ZfKpzyaedH6ajYSiKmvaSwyCs",
                        project_name="rezende", workspace="fgolemo")
experiment.log_parameters(params)
experiment.set_name(exp_name)
experiment.add_tag("cube")

def normal(x, mu, sigma_sq):
    a = (-1 * (x - mu).pow(2) / (2 * sigma_sq)).exp()
    b = 1 / (2 * sigma_sq * pi.expand_as(sigma_sq)).sqrt()

    return a * b


env = Renderer(params["WIDTH"], params["HEIGHT"])

# data_generator = CubeSingleViewGenerator(params["WIDTH"], params["HEIGHT"])
data_generator = RotatingCubeGenerator(params["WIDTH"], params["HEIGHT"])

torch.manual_seed(params["SEED"])
np.random.seed(params["SEED"])

# TODO: the point of this script is to kick out the policy and learn the 160 mus and variances directly,
#  since the shape is fixed and the network doesn't need any input

# TODO: ACTUALLY IMPLEMENT THIS

# TODO, see https://discuss.pytorch.org/t/valueerror-cant-optimize-a-non-leaf-tensor/21751/3

# latent = np.ones(160) * .5
#
# w_ = torch.tensor(1., requires_grad=True)
# 	b_ = torch.tensor(0.1, requires_grad=True)
# 	w = w_.to(device)
# 	b = b_.to(device)
#
#
# 	criteon = nn.MSELoss()
# 	optimizer = optim.Adam([w_, b_], lr=lr)
#
# 	for i in range(500):
# 		x = torch.rand(1).to(device)[0]
#
#
# 		pred = w * x + b
# 		y = 2 * x + 3
# 		loss = criteon(pred, y)
#
# 		grads = torch.autograd.grad(loss, [w_, b_])
# 		w_.grad.fill_(grads[0])
# 		b_.grad.fill_(grads[1])
# 		optimizer.step()


optimizer = torch.optim.Adam(a.parameters(), lr=params["LR"])
eps = np.finfo(np.float32).eps.item()

if not os.path.exists(exp_dir):
    os.mkdir(exp_dir)

import matplotlib.pyplot as plt

for i_episode in trange(params["NUM_EPISODES"]):
    # sample image
    state_raw = npa(data_generator.sample(), dtype=np.float32) / 255

    state = torch.Tensor([np.swapaxes(state_raw, 0, 2)]).to(device)

    # encode to latent variables (mu/var)
    latent_mu, latent_variance = policy.encode(state)

    experiment.log_metric("mu mean", np.mean(latent_mu.detach().view(-1).cpu().numpy()))
    experiment.log_metric("mu min", np.min(latent_mu.detach().view(-1).cpu().numpy()))
    experiment.log_metric("mu max", np.max(latent_mu.detach().view(-1).cpu().numpy()))

    experiment.log_metric("var mean", np.mean(latent_variance.detach().view(-1).cpu().numpy()))
    experiment.log_metric("var min", np.min(latent_variance.detach().view(-1).cpu().numpy()))
    experiment.log_metric("var max", np.max(latent_variance.detach().view(-1).cpu().numpy()))

    # calculate additional VAE loss
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + torch.log(latent_variance) - latent_mu.pow(2) - latent_variance)
    # KLD = -0.5 * torch.sum(1 + latent_logvar - latent_mu.pow(2) - latent_logvar.exp())

    rewards = []
    rewards_raw = []
    log_probs = []
    entropies = []

    for k in range(params["SAMPLES"]):
        # sample K times
        # eps = torch.randn(latent_mu.size()).to(device)
        eps = torch.normal(mean=torch.zeros_like(latent_mu), std=.1).to(device)
        action = torch.sigmoid(latent_mu + latent_variance.sqrt() * eps)
        prob = normal(action, latent_mu, latent_variance)
        log_prob = (prob + 0.0000001).log()
        entropy = -0.5 * ((latent_variance + 2 * pi.expand_as(latent_variance)).log() + 1)

        log_probs.append(log_prob)
        entropies.append(entropy)

        # vertex_params = policy.decode(action).detach().view(-1).cpu().numpy()
        vertex_params = action.detach().view(-1).cpu().numpy()

        experiment.log_metric("vertices mean", np.mean(vertex_params))
        experiment.log_metric("vertices min", np.min(vertex_params))
        experiment.log_metric("vertices max", np.max(vertex_params))

        # render out an image for each of the K samples
        # IMPORTANT THIS CURRENTLY ASSUMES BATCH SIZE = 1
        next_state = env.render(vertex_params, data_generator.cam)
        next_state = npa(next_state, dtype=np.float32) / 255

        # calculate reward for each one of the K samples
        reward_raw = -F.binary_cross_entropy(
            torch.tensor(next_state, requires_grad=False).float(),
            torch.tensor(state_raw, requires_grad=False).float(), reduction='sum')

        rewards_raw.append(reward_raw)

    # deduct average reward of all K-1 samples (variance reduction)
    rewards = npa(rewards_raw) - np.mean(rewards_raw)

    returns = torch.tensor(rewards).float().to(device)

    policy_loss = []
    for log_prob, R, ent in zip(log_probs, returns, entropies):
        # policy_loss.append(-log_prob * R - (0.0001 * ent))
        policy_loss.append(-log_prob * R)

    optimizer.zero_grad()

    policy_loss_sum = torch.cat(policy_loss).sum() / len(policy_loss)
    if i_episode >= params["KLD_DELAY"]:
        policy_loss_sum += params["KLD_WEIGHT"] * KLD

    policy_loss_sum += params["VARIANCE_WEIGHT"] * torch.norm(latent_variance)

    loss_copy = policy_loss_sum.detach().cpu().numpy().copy()
    policy_loss_sum.backward()

    optimizer.step()

    if i_episode % params["CHKP_FREQ"] == 0:
        torch.save(policy.state_dict(), os.path.join(exp_dir, 'reinforce-' + str(i_episode) + '.pkl'))

        img = np.zeros((params["HEIGHT"], params["WIDTH"] * 3, 3), dtype=np.uint8)
        img[:, :params["WIDTH"], :] = np.around(state_raw * 255, 0)
        img[:, params["WIDTH"]:params["WIDTH"]*2, :] = np.around(next_state * 255, 0)
        diff = np.sum(state_raw - next_state, axis=2) + 2 / 4
        diff = np.dstack((diff, diff, diff))
        img[:, params["WIDTH"]*2:, :] = np.around((diff) * 255, 0)
        experiment.log_image(img, name="{:04d}".format(i_episode))

    experiment.log_metric("rewards", np.mean(rewards_raw))
    experiment.log_metric("loss", float(loss_copy))
    experiment.log_metric("kld", KLD.item())
