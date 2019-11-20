import os
from time import strftime
import wandb
import numpy as np
import torch
from torch.nn import functional as F
from tqdm import trange

from threedee_tools.datasets import CubeLoader
from threedee_tools.reinforce import ReinforcePolicy
from threedee_tools.renderer import Renderer
from threedee_tools.utils_3d import t2n

LOGGING = True

params = {
    "SEED": 123,
    "SAMPLES": 10,
    "NUM_EPISODES": 1500,
    "LATENT_SIZE": 160,
    "CHKP_FREQ": 10,  # model saving freq
    "WIDTH": 128,
    "HEIGHT": 128,
    "LR": 1e-5,  # learning rate
    "KLD_WEIGHT": .1,  # learning rate
    "KLD_DELAY": 0,  # how many episodes do we skip adding the KLD to the loss
    "VARIANCE_WEIGHT": 1,
    "NOISE": .1
}

npa = np.array

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pi = torch.Tensor([np.pi]).float().to(device)

exp_name = "30-cube"
exp_dir = "experiments/" + exp_name + "-" + strftime("%Y%m%d%H%M%S")

if LOGGING:
    wandb.init(project="rezende-30", config=params)

# experiment = Experiment(api_key="ZfKpzyaedH6ajYSiKmvaSwyCs",
#                         project_name="rezende", workspace="fgolemo")
# experiment.log_parameters(params)
# experiment.set_name(exp_name)
# experiment.add_tag("v3")


def normal(x, mu, sigma_sq):
    a = (-1 * (x - mu).pow(2) / (2 * sigma_sq)).exp()
    b = 1 / (2 * sigma_sq * pi.expand_as(sigma_sq)).sqrt()
    return a * b


env = Renderer(params["WIDTH"], params["HEIGHT"], shape="ijcv")

data_generator = CubeLoader()

torch.manual_seed(params["SEED"])
np.random.seed(params["SEED"])

policy = ReinforcePolicy(params["LATENT_SIZE"], 160).to(device)
if LOGGING:
    wandb.watch(policy)

optimizer = torch.optim.Adam(policy.parameters(), lr=params["LR"])
eps = np.finfo(np.float32).eps.item()

if not os.path.exists(exp_dir):
    os.mkdir(exp_dir)

fixed_cam = npa([0, 0, 0])

for i_episode in trange(params["NUM_EPISODES"]):
    # sample image
    state = torch.Tensor(data_generator.sample()).permute(2, 0, 1).unsqueeze(0).to(device)
    env.base_light = -data_generator.light + 1

    # encode to latent variables (mu/var)
    latent_mu, latent_variance = policy.encode(state)

    lmu_npy = latent_mu.detach().view(-1).cpu().numpy()
    lva_npy = latent_variance.detach().view(-1).cpu().numpy()

    if LOGGING:
        wandb.log({
            "mu mean": np.mean(lmu_npy),
            "mu min": np.min(lmu_npy),
            "mu max": np.max(lmu_npy),
            "var mean": np.mean(lva_npy),
            "var min": np.min(lva_npy),
            "var max": np.max(lva_npy),
        })

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

        if LOGGING:
            wandb.log({
                "vertices mean": np.mean(vertex_params),
                "vertices min": np.min(vertex_params),
                "vertices max": np.max(vertex_params)
            })

        vertex_params = np.clip(vertex_params, 0, 1)

        # render out an image for each of the K samples
        # IMPORTANT THIS CURRENTLY ASSUMES BATCH SIZE = 1
        next_state = env.render(
            vertex_params, fixed_cam, cam_pos=data_generator.cam + .7)
        next_state = npa(next_state, dtype=np.float32) / 255
        # next_state = make_greyscale(npa(next_state, dtype=np.float32))

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

    policy_loss_sum += params["VARIANCE_WEIGHT"] * torch.norm(latent_variance)

    loss_copy = policy_loss_sum.detach().cpu().numpy().copy()
    policy_loss_sum.backward()

    optimizer.step()

    if i_episode % params["CHKP_FREQ"] == 0:
        torch.save(
            policy.state_dict(),
            os.path.join(exp_dir, 'reinforce-' + str(i_episode) + '.pkl'))

        img = np.zeros((params["HEIGHT"], params["WIDTH"] * 3, 3),
                       dtype=np.uint8)
        img[:, :params["WIDTH"], :] = np.around(t2n(state) * 255, 0)
        img[:, params["WIDTH"]:params["WIDTH"] * 2, :] = np.around(
            next_state * 255, 0)
        diff = (np.sum(t2n(state) - next_state, axis=2) + 3) / 6
        diff = np.dstack((diff, diff, diff))
        img[:, params["WIDTH"] * 2:, :] = np.around((diff) * 255, 0)

        if LOGGING:
            wandb.log({"img": wandb.Image(img, caption="{:04d}".format(i_episode))})

    if LOGGING:
        wandb.log({
            "rewards": np.mean(rewards_raw),
            "loss": float(loss_copy),
            "kld": KLD.item()
        })
