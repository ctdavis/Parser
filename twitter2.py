import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pandas as pd
import re
from itertools import chain
from collections import Counter
from copy import deepcopy
import random

from utils import *
from models import *

def preprocessor(x, lower=True):
    if lower:
        x = x.lower()
    x = re.sub(r'&amp;', 'and', x)
    x = re.sub(r'(http\S+)',r'<u>',x)
    x = re.sub(r'cancelled flight(led)?', r'cancelled', x)
    x = re.sub(r'(.)\1{2,}', r'\1', x)
    x = re.sub(r'(\W)', r' \1 ', x)
    x = re.sub(r'(#|@) (\S)', r'\1\2', x)
    x = re.sub(r'< u >', r'<u>', x)
    x = re.sub(r'\s{2,}', r' ', x)
    x = re.sub(r'(\?) (!)', r'\1\2', x)
    x = re.sub(r'(!) (\?)', r'\1\2', x)
    x = re.sub(r'\.{2,}', r'.', x)
    x = re.sub(r'([.!?]) ', r'\1<@@@>', x)
    return [s.strip().split(' ') for s in x.split('<@@@>') if s.strip().split(' ') != ['']]

def attach_to_leaves(tree, leaves, V, io, G, source):
    leaves = [
        w if w not in V else V[w].replace('[','{').replace(']','}')
        for w in leaves.squeeze(1).log_softmax(-1).argmax(-1).tolist()
    ]
    for leaf_ix, leaf in enumerate(leaves):
        if type(leaf) != str:
            a = leaf - io
            if a < len(source):
                leaves[leaf_ix] = source[a]
            else:
                leaves[leaf_ix] = str(a)

    G.attach_to_leaves(tree, leaves)
    return tree

limit = 10

df = pd.read_csv('data/Tweets.csv')

df = df[df.text.map(preprocessor).map(lambda x: max(map(len, x)) <= limit)].sample(1000)

V = { '<pad>': 0, '<unk>': 1 }
V.update({w:(ix+2) for ix,w in enumerate([k for k,v in Counter(chain(*chain(*df.text.map(preprocessor)))).items() if v > 1])})
rV = {ix:w for w,ix in V.items()}
cV = { '<unk>': 0 }
cV.update({c:(ix+1) for ix,c in enumerate([k for k,v in Counter(chain(*V.keys())).items() if v > 1])})

v_encoder = lambda sentences: [
    torch.LongTensor([1 if w not in V else V[w] for w in s])
    for s in sentences
]

char_v_encoder = lambda sentences: [
    [
        F.one_hot(
            torch.LongTensor([0 if c not in cV else cV[c] for c in w]),
            num_classes=len(cV)
        ).float().unsqueeze(0).transpose(1,2)
        for w in s
    ]
    for s in sentences
]

n_epochs = 100
e = 128
h = 128
act = F.selu

ce = nn.CrossEntropyLoss()
mse = nn.MSELoss()

PAD = 0

model_config = {
    'io': len(V),
    'h': h,
    'act': act,
    'preterminal': False,
    'wd': None,
    'pad': PAD,
    'limit': limit,
    'dr': None,
}

char_config = {
    'io': len(cV),
    'h': h,
    'act': act
}

#classifier_config = deepcopy(model_config)
#classifier_config.update({ 'io': 3, 'n_heads': 1 })
copy_config = deepcopy(model_config)
copy_config.update({ 'n_heads': None })

E = Encoder(**model_config)
G = Generator(**model_config)
C = Copy(**copy_config)
charE = CharEncoder(**char_config)
#CL = Classifier2(**classifier_config)

for m in [E, G, C, charE]: #, CL]:
    m.apply(weight_init)

C.V.weight.data = E.embed.weight

optM = optim.AdamW(chain(E.parameters(), G.parameters(), C.parameters(), charE.parameters()), amsgrad=True)

total_obs = len(list(chain(*df.text.map(preprocessor))))

for e in range(n_epochs):

    epoch_recon_loss = 0.
    epoch_depth_loss = 0.
    epoch_swd = 0.
    epoch_acc = 0.

    for obs in df.text.sample(len(df)):

        obs_preprocessed = preprocessor(obs)
        char_obs_preprocessed = [[list(w) for w in s] for s in obs_preprocessed]

        obs_combo = list(zip(obs_preprocessed, char_obs_preprocessed))

        random.shuffle(obs_combo)

        obs_preprocessed, char_obs_preprocessed = zip(*obs_combo)

        obs_encoded = v_encoder(obs_preprocessed)
        char_obs_encoded = char_v_encoder(char_obs_preprocessed)

        loss = 0.

        optM.zero_grad()

        for s, char_s in zip(obs_encoded, char_obs_encoded):

            target = torch.cat([w.unsqueeze(0) if w != 1 else torch.LongTensor([ix+model_config['io']]) for ix,w in enumerate(s)])

            char_encodings = torch.cat([
                charE(w) for w in char_s
            ])

            features, _ = E(s.unsqueeze(1), char_encodings.unsqueeze(1))
            tree = G(G.act(features.sum(0)), sizes=[len(s)], return_trees=True)[0]
            states = G.get_states(tree).unsqueeze(1).mean(0)
            depth = G.get_leaves(tree, attr='depth').sum(0)
            leaves = G.get_leaves(tree)
            diff = len(features) - len(leaves)
            leaves = torch.cat([leaves, torch.zeros((diff, leaves.shape[-1]))])
            #break
            leaves = C(leaves.unsqueeze(1), G.act(features), sizes=[len(s)])
            #_leaves = leaves.clone()
            #leaves = leaves.view(leaves.shape[0] * leaves.shape[1], -1)
            leaves = leaves.squeeze(1)
            target = target.view(-1)
            #diff = len(target) - len(leaves)
            #leaves = torch.cat([leaves, torch.zeros((diff, leaves.shape[-1]))])

            recon_loss = ce(leaves, target)
            depth_loss = mse(depth, expected_depth([len(s)]))
            swd = sliced_wasserstein_distance(states)

            epoch_recon_loss += recon_loss.item()
            epoch_depth_loss += depth_loss.item()
            epoch_swd += swd.item()
            epoch_acc += (leaves.log_softmax(-1).argmax(-1) == target).long().float().mean().item()

            loss += recon_loss + depth_loss + swd
        #break
        loss.backward()

        torch.nn.utils.clip_grad_norm_(chain(E.parameters(), G.parameters(), C.parameters(), charE.parameters()), 1.)

        optM.step()
    #break
    tree = attach_to_leaves(tree, leaves, rV, model_config['io'], G, obs_preprocessed[-1])

    recon_loss = round(epoch_recon_loss / total_obs, 3)
    depth_loss = round(epoch_depth_loss / total_obs, 3)
    swd = round(epoch_swd / total_obs, 3)
    acc = round(epoch_acc / total_obs, 3) * 100

    print(f'Epoch {e}, loss / depth / swd / acc : {recon_loss} / {depth_loss} / {swd} / {acc}\n')
    print(f'Reconstruction of: {" ".join(obs_preprocessed[-1])}')
    print(f'                   {" ".join([rV[w] for w in obs_encoded[-1].tolist()])}')
    print_tree(tree, lambda x: x, attr='attachment')

