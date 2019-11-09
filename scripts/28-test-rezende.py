import os
import numpy as np
import torch
from threedee_tools.reinforce import ReinforcePolicy
from threedee_tools.trained_models import MODEL_PATH
from threediqtt.dataset import ValDataset
import torch.nn.functional as F
from tqdm import tqdm

policy = ReinforcePolicy(160, 160)
# policy.load_state_dict(torch.load(MODEL_PATH + "/s0/reinforce-0.pkl")) # 52.17
# policy.load_state_dict(torch.load(MODEL_PATH + "/s0/reinforce-1.pkl")) # 50.88
# policy.load_state_dict(torch.load(MODEL_PATH + "/s0/reinforce-2.pkl")) # 53.02%
# (52.17 + 50.88 + 53.02) / 3 = 52.02%

# policy.load_state_dict(torch.load(MODEL_PATH + "/s200/reinforce-0.pkl")) # 58.379999999999995%
# policy.load_state_dict(torch.load(MODEL_PATH + "/s200/reinforce-1.pkl")) # 63.82%
# policy.load_state_dict(torch.load(MODEL_PATH + "/s200/reinforce-2.pkl")) # 62.45%
# (58.379 + 63.82 + 62.45) / 3 = 61.55%

# policy.load_state_dict(torch.load(MODEL_PATH + "/s1000/reinforce-0.pkl")) # 67.78%
# policy.load_state_dict(torch.load(MODEL_PATH + "/s1000/reinforce-1.pkl")) # 70.89999999999999%
# policy.load_state_dict(torch.load(MODEL_PATH + "/s1000/reinforce-2.pkl")) # 71.37%
# (67.78 + 70.899 + 71.37) / 3 = 70.01 %

# policy.load_state_dict(torch.load(MODEL_PATH + "/s10000/reinforce-0.pkl")) # 71.97%
# policy.load_state_dict(torch.load(MODEL_PATH + "/s10000/reinforce-1.pkl")) # 78.64%
# policy.load_state_dict(torch.load(MODEL_PATH + "/s10000/reinforce-2.pkl")) # 79.47%
# (71.97 + 78.64 + 79.47) / 3 =  76.69 %

policy.eval()

dataset = ValDataset(os.path.expanduser("/Volumes/dell/3diqtt-v2-val.h5"))

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