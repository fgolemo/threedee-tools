import os
import numpy as np
import torch
from threedee_tools.reinforce import ReinforcePolicy
from threedee_tools.trained_models import MODEL_PATH
from threediqtt.dataset import ValDataset
import torch.nn.functional as F
from tqdm import tqdm

policy = ReinforcePolicy(160, 160)
policy.load_state_dict(torch.load(MODEL_PATH + "/reinforce-iqtt2.pkl"))
policy.eval()

dataset = ValDataset(os.path.expanduser("~/Downloads/3diqtt-v2-val.h5"))

correct = 0

for idx, item in enumerate(tqdm(dataset)):
    # embed reference img
    ref_mu, _ = policy.encode(item["question"][0].unsqueeze(0))

    # embed all other images
    ans_a, _ = policy.encode(item["question"][1].unsqueeze(0))
    ans_b, _ = policy.encode(item["question"][2].unsqueeze(0))
    ans_c, _ = policy.encode(item["question"][3].unsqueeze(0))

    # rank the other imgs
    diffs = []
    for ans in [ans_a, ans_b, ans_c]:
        diffs.append(F.mse_loss(ans, ref_mu).item())

    answer = np.argmin(diffs)

    # compare ranking with correct answer
    # print ("answer: {}, correct: {}".format(answer, item["answer"][0]))
    if answer == item["answer"][0]:
        correct += 1

    if idx == len(dataset)-1:
        break

print ("correct: {}%".format(correct/len(dataset)*100))