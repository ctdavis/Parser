import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pytorch_ranger import Ranger
from fast_pytorch_kmeans import KMeans
import pandas as pd
import numpy as np
import random
from itertools import chain
from functools import partial
from copy import deepcopy
import nltk

import torch.autograd.profiler as profiler

from utils import *
from models import *
from get_wiki_data import *

PATH = 'random_wiki_02_20_2021'

limit = 30

download_new_data = False
download_new_vocab = False

if download_new_data is True:
    _, df = get_vocab(preprocessor, n_samples=3000)
else:
    with open(f'{PATH}/df.pkl', 'rb') as f:
        df = pickle.load(f)

if download_new_vocab is True:
    ds, _ = get_vocab(preprocessor, n_samples=1000)
else:
    with open(f'{PATH}/V.pkl', 'rb') as f:
        V = pickle.load(f)
    with open(f'{PATH}/cV.pkl', 'rb') as f:
        cV = pickle.load(f)
    with open(f'{PATH}/rV.pkl', 'rb') as f:
        rV = pickle.load(f)
        
    ds = Dataset({ 
        'df': df,
        'limit': limit,
        'vrs': { 
            'text': { 
                'reference': 'text', 
                'vocab_min_freq': 1, 
                'char_min_freq': 1, 
                'rank': 0.05,
                'preprocessor': preprocessor,
                'base_vocab': {},
            }
        }
    })
    ds.vrs['text']['V'] = V
    ds.vrs['text']['cV'] = cV
    ds.vrs['text']['rV'] = rV
    ds.vrs['text']['preprocessor'] = preprocessor
    #ds.df = None
    
model_config = {
    'i': len(ds.vrs['text']['V']),
    'o': len(ds.vrs['text']['V']),
    'char_i': len(ds.vrs['text']['cV']),
    'h': 256,
    'm': 4,
    'reduction': 16,
    'n_heads': 16, # try > 1 heads
    'n_copy_heads': 16, # try 2 heads
    'limit': limit + 1, # add 1 to allow for prompt based prediction (i.e. cat -> animal, dog -> animal, car -> ~animal, eel -> [MASK])
    'add_limit_to_i': False,
    'long_tensor_input': True,
    'wd': None,
    'n_slots': 8,
    'discretize': 0,
    'span_dropout': None, #span_dropout,
}

model_config.update({ 'char_kwargs': deepcopy(model_config) })
model_config['char_kwargs']['i'] = model_config['char_i']
model_config['char_kwargs']['o'] = model_config['char_i']
model_config['char_kwargs']['wd'] = None
model_config['char_kwargs']['discretize'] = 0
model_config['char_kwargs']['char_level'] = True
model_config['char_kwargs']['n_heads'] = 1

P = Parser(**model_config)

opt = Ranger(P.parameters())

mse = nn.MSELoss()
ce = nn.CrossEntropyLoss()

data = pd.DataFrame({'text': list(filter(
    lambda x: (lambda y: y != [] and len(y[0]) <= limit)(preprocessor(x)),
    chain(*df['text'].apply(nltk.sent_tokenize).tolist())
))})
data = data.sample(len(data))

n_sentences = len(data)

print(f'Size: {n_sentences}, Vocab: {len(ds.vrs["text"]["V"])}')

freq = 100
kmeans = [KMeans(n_clusters=i, mode='euclidean') for i in range(2, 6)]
n_obs = 0

for e in range(n_sentences):

    (_d, c, d) = ds.preprocess(data.iloc[e:e+1], ['text'], flatten=True)['text'][0]

    P.train()
    if e == 0 or e % freq == 1:
        epoch_recon = 0.
        epoch_depth = 0.
        epoch_swd = 0.
        epoch_acc = 0.
        epoch_char_loss = 0.
        epoch_char_acc = 0.
        examples = []
        epoch_cat = []

    # update model
    (recon_loss, depth_loss, swd_loss, acc), (char_loss, char_depth, char_swd, char_acc) = compute_loss_and_acc(
        P, _d[0], c[0], model_config, ce=ce, mse=mse, discretization=model_config['discretize'], size=len(_d[0]), char_level=True
    )
    loss = (recon_loss + depth_loss + swd_loss + char_loss + char_depth + char_swd)
    #"""
    if random.choice([True] + [False]): # consistency loss
        states0 = (lambda x: torch.cat([P.decoder.get_states(x), P.decoder.get_states(x, 'h')], dim=1))(P(_d[0], c[0])['tree'])
        states1 = (lambda x: torch.cat([P.decoder.get_states(x), P.decoder.get_states(x, 'h')], dim=1))(P(_d[0], c[0])['tree'])
        if len(states0) < len(states1):
            states0 = torch.cat([states0, torch.zeros((len(states1) - len(states0), states0.shape[-1]))])
        elif len(states1) < len(states0):
            states1 = torch.cat([states1, torch.zeros((len(states0) - len(states1), states1.shape[-1]))])
        loss += mse(states0, states1)
    #"""
    #"""
    if random.choice([True] + [False] * 4): # clustering loss
        random_sample = data.sample(50).text.tolist()                    
        encodings = [
            st['state']
            for d_ in random_sample
            for st in P.decoder.get_subtrees(parse_sentence(d_, ds, P, model_config, _print_tree=False), len(preprocessor(d_)[0]))
        ]            

        encodings = random.sample(encodings, 50)

        z = torch.cat(encodings).mean(0, keepdim=True)
        cluster_ids = random.choice(kmeans).fit_predict(torch.cat(encodings).detach())
        cat_dataset_ = list(zip(encodings, cluster_ids))
        cat_dataset = []
        for cat in cluster_ids.unique().tolist():
            cat_dataset += [(lambda y: [[z[0] for z in y[:-1]], y[-1][0]])(list(filter(lambda x: x[1] == cat, cat_dataset_)))]
        cat_dataset = list(filter(lambda x: len(x[0]) > 1, cat_dataset))
        cat_losses = []
        for support, query in cat_dataset:
            if random.choice([True, False]):
                query = random.choice(encodings)
                target = torch.zeros((1,)).long()
            else:
                target = torch.ones((1,)).long()
            z_dist = -(z - query).pow(2).sum().view(1,1)
            s_dist = -(torch.cat(support).mean(0, keepdim=True) - query).pow(2).sum().view(1,1)
            dists = torch.cat([z_dist, s_dist]).view(1,2)
            cat_losses += [ce(dists, target).view(1,1)]
        cat_loss = torch.cat(cat_losses).mean() * 1.
        epoch_cat += [cat_loss.item()]
        loss += cat_loss
    #"""
    loss.backward()
    opt.step()
    opt.zero_grad()

    # record progress
    epoch_recon += recon_loss.item()
    epoch_depth += depth_loss.item()
    epoch_swd += swd_loss.item()
    epoch_acc += acc
    epoch_char_loss += char_loss.item()
    epoch_char_acc += char_acc

    examples += [[_d, c]]

    if e != 0 and e % freq == 0:

        P.eval()
        recon = round(epoch_recon / freq, 3)
        depth = round(epoch_depth / freq, 3)
        swd = round(epoch_swd / freq, 3)
        acc = round(epoch_acc / freq, 2) * 100
        char_loss = round(epoch_char_loss / freq, 3)
        char_acc = round(epoch_char_acc / freq, 2) * 100
        cl = round(sum(epoch_cat) / len(epoch_cat), 3)
        
        test_dict = P(_d[0], c[0])
        tree, leaves = test_dict['tree'], test_dict['leaves_after_copy']
        print(f'epoch: {e // freq} / loss: {recon} / depth: {depth} / swd: {swd} / acc: {acc} / closs: {char_loss} / cacc: {char_acc} / cl: {cl}')
        print(f'reconstruction of: {" ".join(d[0])}')
        try:
            parse_sentence(" ".join(d[0]), ds, P, model_config)
        except:
            print('parse failed')

