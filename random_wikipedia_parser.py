import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchcontrib.nn as tcnn
import pandas as pd
import re
from functools import partial
from itertools import chain
import json
from copy import deepcopy
import math

from utils import *
from models import *

r0 = lambda x: re.sub(r'([A-Za-z])([^A-Za-z])([A-Za-z])', r'\1 \2 \3', x)  
r1 = lambda x: re.sub(r'([A-Za-z])([^A-Za-z])', r'\1 \2', x)
r2 = lambda x: re.sub(r'([^A-Za-z])([A-Za-z])', r'\1 \2', x)
r3 = lambda x: re.sub(r'([$._-]*[0-9][0-9.,$_:-]*)', r'<n>', x)
r4 = lambda x: re.sub(r'(\[<n>\])+', r'', x)
r5 = lambda x: re.sub(r'^\s*coordinates :.*$', r'', x)
r6 = lambda x: re.sub(r'\n', r' ', x)
r7 = lambda x: re.sub(r'\s{2,}', r' ', x)
preprocessor = lambda x: [w for w in r7(r6(r5(r4(r3(r2(r1(r0(x.lower())))))))).strip().split(' ') if w != '']

limit = 10
config = {
    'limit': limit,
    'df': 'random_wiki',
    'anchor': 'text',
    'sample_size': 100,
    'test_size': 10,
    'unify': [], #'text','chars'],
    'sleep': 0.5,
    'vars': {
        'text': {
            'preprocessor': preprocessor,
            'limit': 1,
            'pad': True,
            'extra_fxns': {
                'sizes': ('text', lambda context, x: len(x))
             },
        },
        'chars': {
            'source': 'text',
            'preprocessor': lambda x: list(map(list, preprocessor(x))),
            'limit': 2,
            'pad': True,
            'extra_fxns': {
                'sizes': ('text', lambda context, x: list(map(len, x)))
            }
        },
    }
}

ds = Dataset(config)

PAD = 0
SD = 1.

si = partial(shuffle_indices, list(range(ds.sample_size))) # returns shuffled list of indices

batch_size = 32
n_epochs = 200
n_batches = ds.sample_size // batch_size
e = 128
h = 128
act = F.selu
max_char_size = max(map(len, chain(*ds.vars['chars']['text'])))
start_char_modelling_at = 0.0

ce = nn.CrossEntropyLoss()
mse = nn.MSELoss()

model_config = {
    'io': len(ds.vars['text']['vocab']),
    'h': h,
    'batch': batch_size,
    'act': act,
    'preterminal': False,
    'wd': None,
    'pad': PAD,
    'limit': limit,
}

char_model_config = deepcopy(model_config)
char_model_config.update({ 'wd': None, 'h': 32, 'adaptor': h, 'has_subsequence': False, 'io': len(ds.vars['chars']['vocab']) })

charE = Encoder(**char_model_config)
E = Encoder(**model_config)
G = Generator(**model_config)
C = Copy(**model_config)

charE.apply(weight_init)
E.apply(weight_init)
G.apply(weight_init)
C.apply(weight_init)

C.V.weight.data = E.embed.weight

optM = optim.AdamW(chain(E.parameters(), G.parameters(), charE.parameters(), C.parameters()))

for e in range(n_epochs):

    ixs = si()
    epoch_gloss = 0.
    epoch_gacc = 0.
    epoch_depth_loss = 0.
    epoch_swd = 0.

    E.train()
    G.train()
    charE.train()
    C.train()

    ###########
    #  train  #
    ###########

    for ix, b in enumerate(batch_indices(ixs, batch_size, n_batches)): 

        optM.zero_grad()

        T, Tchars, sizes = zip(*[
            (ds.vars['text']['vectors'][i], ds.vars['chars']['vectors'][i], ds.vars['text']['sizes'][i])
            for i in b
        ])

        Targ = [torch.cat([w.unsqueeze(0) if w != 1 else torch.LongTensor([ix+model_config['io']]) for ix,w in enumerate(t)]) for t in T]

        _Tchars = batch_data([torch.cat([charE(w.unsqueeze(1)) for w in s]) for s in Tchars], PAD, limit, h)
        T = batch_data(list(T), PAD, limit)
        Targ = batch_data(list(Targ), PAD, limit)

        features = E(T, _Tchars)

        trees = G(act(features.sum(0)), sizes=[limit] * len(sizes), return_trees=True)

        leaves, depths, states = zip(*[
            (
                G.get_leaves(t),
                G.get_leaves(t,attr='depth'), 
                G.get_states(t).unsqueeze(1).mean(0)
            )
            for t in trees
        ])

        depths = torch.cat([d.sum(0,keepdim=True) for d in depths]).squeeze(1)
        states = torch.cat(states)

        leaves, Tg = pad(define_padded_vectors(nn.utils.rnn.pad_sequence(leaves), PAD), Targ)
        leaves = C(leaves, features, sizes)

        leaves = leaves.view(leaves.shape[0] * leaves.shape[1], -1)
        Tg = Tg.contiguous().view(-1)

        gloss = ce(leaves, Tg)
        depth_loss = mse(depths, expected_depth(sizes))
        swd = sliced_wasserstein_distance(states)

        loss = gloss + depth_loss + swd

        loss.backward()

        optM.step()

        epoch_gloss += gloss.item()
        epoch_depth_loss += depth_loss.item()
        epoch_swd += swd.item()
        epoch_gacc += (leaves.log_softmax(-1).argmax(-1) == Tg).long().float().mean().item()

    ##########
    #  test  #
    ##########

    E.eval()
    G.eval()
    charE.eval()
    C.eval()

    test_words_data = ds.preprocess_new_observations('text', ds.test_df)
    test_chars_data = ds.preprocess_new_observations('chars', ds.test_df)

    test_words_targ = [
        torch.cat([w.unsqueeze(0) if w != 1 else torch.LongTensor([ix+model_config['io']]) for ix,w in enumerate(t)])
        for t in test_words_data['vectors']
    ]
        
    test_chars = batch_data([torch.cat([charE(w.unsqueeze(1)) for w in s]) for s in test_chars_data['vectors']], PAD, limit, h)
    test_words = batch_data(list(test_words_data['vectors']), PAD, limit)
    test_words_targ = batch_data(list(test_words_targ), PAD, limit)

    test_features = E(test_words, test_chars)

    test_trees = G(act(test_features.sum(0)), sizes=[limit] * len(test_words_data['sizes']), return_trees=True)

    tleaves, tdepths, tstates = zip(*[
        (
            G.get_leaves(t),
            G.get_leaves(t,attr='depth'), 
            G.get_states(t).unsqueeze(1).mean(0)
        )
        for t in test_trees
    ])

    tdepths = torch.cat([d.sum(0,keepdim=True) for d in tdepths]).squeeze(1)
    tstates = torch.cat(tstates)

    tleaves, test_words_targ = pad(define_padded_vectors(nn.utils.rnn.pad_sequence(leaves), PAD), test_words_targ)
    tleaves = C(tleaves, test_features, [limit]*model_config['test_size'])

    tleaves = tleaves.view(tleaves.shape[0] * tleaves.shape[1], -1)
    test_words_targ = test_words_targ.contiguous().view(-1)

    tgloss = ce(tleaves, test_words_targ)
    tdepth_loss = mse(tdepths, expected_depth(sizes))
    tswd = sliced_wasserstein_distance(tstates)

    #####################################
    #  output parsed training sentence  #
    #####################################

    features = E(T[:, :1], _Tchars[:, :1])
    trees = G(act(features.sum(0)), sizes=[limit], return_trees=True)
    leaves, _ = pad(G.get_leaves(trees[0]).unsqueeze(1), T[:,:1])
    leaves = C(leaves, features, sizes[:1])

    tree = attach_to_leaves(trees[0], leaves, ds.vars['text'], model_config['io'], G, ds.vars['text']['text'][b[0]])

    print(f'Epoch: {e+1}')
    print(f'GLoss: {round(epoch_gloss / n_batches, 3)}')
    print(f'DepthLoss: {round(epoch_depth_loss / n_batches, 3)}')
    print(f'SWD: {round(epoch_swd / n_batches, 3)}')
    print(f'GAcc: {round(epoch_gacc / n_batches, 3)}')
    print(f'Test loss/depth/swd: {tgloss.item(), tdepth_loss.item(), tswd.item()}')
    print(f'Reconstruction of: {" ".join([w for w in ds.vars["text"]["text"][b[0]]])}')
    print(f'                   {" ".join([ds.vars["text"]["reverse_vocab"][w.item()] for w in T[:,0] if w.item() != 0])}')
    try:
        print_tree(
            tree,
            lambda x: x,
            attr='attachment'
        )
    except Exception as e:
        print(e)
    print()
