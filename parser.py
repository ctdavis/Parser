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
        epoch_ucl = []
        epoch_uca = []
        examples = []
        epoch_cat = []
        epoch_con = []

    target = create_target(_d[0], model_config['o'])
    parse_dict = P(_d[0], c[0], use_wd=model_config['wd'], size=len(_d[0]), gen_chars=True)

    wrl = word_reconstruction_loss(parse_dict, target)
    wdl = word_depth_loss(parse_dict, target, coeff=0.001)
    wswd = word_swd(parse_dict)
    wacc = word_accuracy(parse_dict, target)
    
    crl = char_reconstruction_loss(parse_dict, c[0])
    cdl = char_depth_loss(parse_dict, c[0], coeff=0.001)
    cswd = char_swd(parse_dict)
    cacc = char_accuracy(parse_dict, c[0])
    
    loss = (wrl + wdl + wswd + crl + cdl + cswd)
    
    if False: #random.choice([True] + [False] * 3):
        n_shots = random.sample(list(range(3, 11)))
        ucl, uca = unsupervised_clustering_loss(data.text, ds, P, model_config, random.choice(kmeans), sample_size=50, n_shots=n_shots) 
        ucl = ucl * 0.1
        loss += ucl
        epoch_ucl += [ucl.item()]
        epoch_uca += [uca]
    
    loss.backward()
    opt.step()
    opt.zero_grad()

    # record progress
    epoch_recon += wrl.item()
    epoch_depth += wdl.item()
    epoch_swd += wswd.item()
    epoch_acc += wacc
    epoch_char_loss += crl.item()
    epoch_char_acc += cacc
    #epoch_ucl += ucl.item()
    #epoch_uca += uca
    #epoch_con += [con.item()]
    
    examples += [[_d, c]]

    if e != 0 and e % freq == 0:

        P.eval()
        recon = round(epoch_recon / freq, 3)
        depth = round(epoch_depth / freq, 3)
        swd = round(epoch_swd / freq, 3)
        acc = round(epoch_acc / freq, 2) * 100
        char_loss = round(epoch_char_loss / freq, 3)
        char_acc = round(epoch_char_acc / freq, 2) * 100
        if epoch_ucl:
            ucl = round(sum(epoch_ucl) / len(epoch_ucl), 3)
            uca = round(sum(epoch_uca) / len(epoch_uca), 2) * 100
        else:
            ucl, uca = 0., 0.
        #cl = 0.#round(sum(epoch_cat) / len(epoch_cat), 3)
        #con = round(sum(epoch_con) / len(epoch_con), 3)
        
        test_dict = P(_d[0], c[0])
        tree, leaves = test_dict['tree'], test_dict['leaves_after_copy']
        print(f'epoch: {e // freq} / loss: {recon} / depth: {depth} / swd: {swd} / acc: {acc} / closs: {char_loss} / cacc: {char_acc} / ucl: {ucl} / uca: {uca}')# / con: {con}')
        print(f'reconstruction of: {" ".join(d[0])}')
        try:
            parse_sentence(" ".join(d[0]), ds, P, model_config)
        except:
            print('parse failed')

