from comet_ml import Experiment
import os
from time import strftime

import numpy as np
import torch
from threedee_tools.utils_rl import ContrastiveLoss
from torch.nn import functional as F
from tqdm import trange, tqdm

from threedee_tools.datasets import IQTTLoader
from threedee_tools.reinforce import ReinforcePolicy
from threedee_tools.renderer import Renderer
from threedee_tools.utils_3d import make_greyscale, t2n

params = {
    "SEED": 123,
    "SAMPLES": 10,
    "NUM_EPISODES": 3333,
    "LATENT_SIZE": 160,
    "CHKP_FREQ": 10,  # model saving freq
    "WIDTH": 128,
    "HEIGHT": 128,
    "LR": 1e-5,  # learning rate
    "KLD_WEIGHT": .1,  # learning rate
    "KLD_DELAY": 100,  # how many episodes do we skip adding the KLD to the loss
    "VARIANCE_WEIGHT": 1,
    "NOISE": .1,
    "SUPERVISED_SAMPLES": 3333,
    "SUPERVISED_WEIGHT": 100
}

supervised_sample_idxs = [int(round(x)) for x in np.linspace(
    start=0, stop=params["NUM_EPISODES"], num=params["SUPERVISED_SAMPLES"])]

print(supervised_sample_idxs)

npa = np.array

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pi = torch.Tensor([np.pi]).float().to(device)

exp_name = "29-iqtt"
exp_dir = "experiments/" + exp_name + "-" + strftime("%Y%m%d%H%M%S")

experiment = Experiment(
    api_key="ZfKpzyaedH6ajYSiKmvaSwyCs",
    project_name="rezende",
    workspace="fgolemo")
experiment.log_parameters(params)
experiment.set_name(exp_name)
experiment.add_tag("v4-{}".format(params["SUPERVISED_SAMPLES"]))


def normal(x, mu, sigma_sq):
    a = (-1 * (x - mu).pow(2) / (2 * sigma_sq)).exp()
    b = 1 / (2 * sigma_sq * pi.expand_as(sigma_sq)).sqrt()
    return a * b


env = Renderer(params["WIDTH"], params["HEIGHT"], shape="iqtt")

data_generator = IQTTLoader(greyscale=True)

torch.manual_seed(params["SEED"])
np.random.seed(params["SEED"])

policy = ReinforcePolicy(params["LATENT_SIZE"], 160).to(device)

contrast_loss = ContrastiveLoss()

optimizer = torch.optim.Adam(policy.parameters(), lr=params["LR"])
eps = np.finfo(np.float32).eps.item()

if not os.path.exists(exp_dir):
    os.mkdir(exp_dir)

fixed_cam = npa([0, 0, 0])

for i_episode in trange(params["NUM_EPISODES"]):
    # sample image
    state_q, state_a, state_d1, state_d2 = data_generator.sample_qa()
    state = state_q.unsqueeze(0).to(device)

    # encode to latent variables (mu/var)
    latent_mu, latent_variance = policy.encode(state)

    experiment.log_metric("mu mean",
                          np.mean(latent_mu.detach().view(-1).cpu().numpy()))
    experiment.log_metric("mu min",
                          np.min(latent_mu.detach().view(-1).cpu().numpy()))
    experiment.log_metric("mu max",
                          np.max(latent_mu.detach().view(-1).cpu().numpy()))

    experiment.log_metric(
        "var mean", np.mean(latent_variance.detach().view(-1).cpu().numpy()))
    experiment.log_metric(
        "var min", np.min(latent_variance.detach().view(-1).cpu().numpy()))
    experiment.log_metric(
        "var max", np.max(latent_variance.detach().view(-1).cpu().numpy()))

    # calculate additional VAE loss
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + torch.log(latent_variance) - latent_mu.pow(2) -
                           latent_variance)
    # KLD = -0.5 * torch.sum(1 + latent_logvar - latent_mu.pow(2) - latent_logvar.exp())

    rewards = []
    rewards_raw = []
    log_probs = []
    entropies = []

    for k in range(params["SAMPLES"]):
        # sample K times
        # eps = torch.randn(latent_mu.size()).to(device)
        eps = torch.normal(
            mean=torch.zeros_like(latent_mu), std=params["NOISE"]).to(device)
        action = latent_mu + latent_variance.sqrt() * eps
        prob = normal(action, latent_mu, latent_variance)
        log_prob = (prob + 0.0000001).log()
        entropy = -0.5 * (
            (latent_variance + 2 * pi.expand_as(latent_variance)).log() + 1)

        log_probs.append(log_prob)
        entropies.append(entropy)

        # vertex_params = policy.decode(action).detach().view(-1).cpu().numpy()
        vertex_params = action.detach().view(-1).cpu().numpy()

        experiment.log_metric("vertices mean", np.mean(vertex_params))
        experiment.log_metric("vertices min", np.min(vertex_params))
        experiment.log_metric("vertices max", np.max(vertex_params))

        vertex_params = np.clip(vertex_params, 0, 1)

        # render out an image for each of the K samples
        # IMPORTANT THIS CURRENTLY ASSUMES BATCH SIZE = 1
        next_state = env.render(vertex_params, fixed_cam)
        next_state = make_greyscale(npa(next_state, dtype=np.float32))

        # calculate reward for each one of the K samples
        reward_raw = -F.binary_cross_entropy(
            torch.tensor(next_state, requires_grad=False).permute(2, 0,
                                                                  1).float(),
            state[0, :, :, :].float(),
            reduction='sum')

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

    if i_episode in supervised_sample_idxs:
        tqdm.write("doing a supervised loopdy loop")

        state_a = state_a.unsqueeze(0).to(device)
        latent_mu_a, _ = policy.encode(state_a)
        policy_loss_sum += params["SUPERVISED_WEIGHT"] * contrast_loss(latent_mu, latent_mu_a, 0)

        state_d1 = state_d1.unsqueeze(0).to(device)
        latent_mu_d1, _ = policy.encode(state_d1)
        policy_loss_sum += params["SUPERVISED_WEIGHT"] * contrast_loss(latent_mu, latent_mu_d1, 1)

        state_d2 = state_d2.unsqueeze(0).to(device)
        latent_mu_d2, _ = policy.encode(state_d2)
        policy_loss_sum += params["SUPERVISED_WEIGHT"] * contrast_loss(latent_mu, latent_mu_d2, 1)


    policy_loss_sum += params["VARIANCE_WEIGHT"] * torch.norm(latent_variance)

    loss_copy = policy_loss_sum.detach().cpu().numpy().copy()
    policy_loss_sum.backward()

    optimizer.step()

    if i_episode % params["CHKP_FREQ"] == 0:
        torch.save(
            policy.state_dict(),
            os.path.join(exp_dir, 'reinforce.pkl'))

        img = np.zeros((params["HEIGHT"], params["WIDTH"] * 3, 3),
                       dtype=np.uint8)
        img[:, :params["WIDTH"], :] = np.around(t2n(state) * 255, 0)
        img[:, params["WIDTH"]:params["WIDTH"] * 2, :] = np.around(
            next_state * 255, 0)
        diff = (np.sum(t2n(state) - next_state, axis=2) + 3) / 6
        diff = np.dstack((diff, diff, diff))
        img[:, params["WIDTH"] * 2:, :] = np.around((diff) * 255, 0)
        experiment.log_image(img, name="{:04d}".format(i_episode))

    experiment.log_metric("rewards", np.mean(rewards_raw))
    experiment.log_metric("loss", float(loss_copy))
    experiment.log_metric("kld", KLD.item())
