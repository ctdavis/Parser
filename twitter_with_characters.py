import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchcontrib.nn as tcnn
import torchcontrib
import pandas as pd
import re
from functools import partial
from itertools import chain
import json
from copy import deepcopy
import math
import pickle
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
    return x.strip().split(' ')

def preprocess_sentiment(x):
    return [x]

def preprocess_chars(x):
    return [list(w) for w in preprocessor(x)]

def size_fxn(context, x):
    return len(x)

def target_fxn(context, x):
    return torch.LongTensor([len(context['vocab']) - (x + 1)])

def char_size_fxn(context, x):
    return list(map(len, x))

def swish(x):
    return x * torch.sigmoid(x)

def mish(x):
    return x * torch.tanh(F.softplus(x))

SAVE_DIR = './saved_models/06_14_01'

limit = 10
config = {
    'len_limit': limit,
    'sample_size': 1000,
    'test_size': 128,
    'source': pd.read_csv('data/Tweets.csv'),
    'unify': ['text','airline_sentiment'], # unify vocabularies of each of these variables
    'reserve': { 'text': ['positive','neutral','negative'] }, # helpful for use in target_fxn (below); ensures these words have the highest indices
    'anchor': 'text',
    'vars': {
        'text': {
            'limit': 4, # retain a given word in vocab only if it occurs a number of times > limit
            'pad': True,
            'preprocessor': preprocessor,
            'extra_fxns': {
                'sizes': ('text', size_fxn) # create additional variable based on lengths of sentences in text
             },
        },
        'chars': {
            'source': 'text',
            'limit': 1, # retain a given word in vocab only if it occurs a number of times > limit
            'pad': True,
            'preprocessor': preprocess_chars,
            'extra_fxns': {
                'sizes': ('text', char_size_fxn) # create additional variable based on lengths of sentences in text
             },
        },
        'airline_sentiment': {
            'limit': 0,
            'pad': False,
            'preprocessor': preprocess_sentiment,
            'extra_fxns': {
                'target': ('vectors', target_fxn) # create additional variable based on index of sentiment label
            }
        }
    }
}

ds = LanguageDataset(config)
torch.save(ds.state_dict(), f'{SAVE_DIR}/data.pt')

output_set = torch.LongTensor([ # used as query vector in attention-based classifier
    ds.vars['text']['vocab']['positive'],
    ds.vars['text']['vocab']['neutral'],
    ds.vars['text']['vocab']['negative']
])

PAD = 0
SD = 1.

train_sample_size = ds.sample_size - config['test_size']

si = partial(shuffle_indices, list(range(train_sample_size))) # returns shuffled list of indices
tsi = partial(shuffle_indices, list(range(config['test_size'])))

batch_size = 16
n_epochs = 200
n_batches = train_sample_size // batch_size
tn_batches = config['test_size'] // batch_size
e = 128
h = 128
act = F.selu

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
    'sentiment': len(ds.vars['airline_sentiment']['vocab']),
    'limit': limit,
    'dr': None,
}

char_config = {
    'io': len(ds.vars['chars']['vocab']),
    'h': h,
    'act': act
}

classifier_config = deepcopy(model_config)
classifier_config.update({ 'io': 3, 'n_heads': 1 })
copy_config = deepcopy(model_config)
copy_config.update({ 'n_heads': None })

with open(f'{SAVE_DIR}/model_configs.pkl', 'wb') as f:
    pickle.dump({ 'model': model_config, 'classifier': classifier_config, 'copy': copy_config, 'char': char_config }, f)

charE = CharEncoder(**char_config)
E = Encoder(**model_config)
G = Generator(**model_config)
C = Copy(**copy_config) # attention based module for choosing to either produce a vocab item or select word for source sentence
CL = Classifier2(**classifier_config) # attention based module for classifying sentiment

charE.apply(weight_init)
E.apply(weight_init)
G.apply(weight_init)
C.apply(weight_init)
CL.apply(weight_init)

C.V.weight.data = E.embed.weight

optM = optim.AdamW(chain(E.parameters(), G.parameters(), C.parameters(), CL.parameters(), charE.parameters()), amsgrad=True)

for e in range(n_epochs):

    ixs = si()
    tixs = tsi()
    epoch_gloss = 0.
    epoch_gacc = 0.
    epoch_depth_loss = 0.
    epoch_swd = 0.
    epoch_cl_loss = 0.
    epoch_cl_acc = 0.

    charE.train()
    E.train()
    G.train()
    C.train()
    CL.train()

    for ix, b in enumerate(batch_indices(ixs, batch_size, n_batches)): 

        T, sizes, sentiment_target, charT, char_sizes = zip(*[
            (
                ds.vars['text']['vectors'][i],
                ds.vars['text']['sizes'][i],
                ds.vars['airline_sentiment']['target'][i],
                ds.vars['chars']['vectors'][i],
                ds.vars['chars']['sizes'][i]
            )
            for i in b
        ])
        Targ = [torch.cat([w.unsqueeze(0) if w != 1 else torch.LongTensor([ix+model_config['io']]) for ix,w in enumerate(t)]) for t in T]
        sentiment_target = torch.cat(sentiment_target)

        char_encodings = batch_data(
            [
                torch.cat([
                    charE(F.one_hot(w,num_classes=len(ds.vars['chars']['vocab']))\
                        .float().unsqueeze(0).transpose(1,2))
                    for w in s
                ])
                for s in charT
            ],
            None,
            limit,
            use_pad_var=False
        )

        T = batch_data(list(T), PAD, limit)
        Targ = batch_data(list(Targ), PAD, limit)

        optM.zero_grad()

        features, _ = E(T, char_encodings) # encode sentences

        trees = G(act(features.sum(0)), sizes=[limit] * len(sizes), return_trees=True) # decode sentences as syntax trees

        leaves, depths, states = zip(*[
            (
                G.get_leaves(t),
                G.get_leaves(t,attr='depth'), 
                G.get_states(t).unsqueeze(1).mean(0)
            )
            for t in trees
        ])

        # classify sentiments based on subtrees
        sentiment, _ = CL([G.get_states(t).unsqueeze(1) for t in trees], [G.get_states(t).unsqueeze(1) for t in trees]) 
        #act(E.embed(output_set)).unsqueeze(1)) 

        depths = torch.cat([d.sum(0,keepdim=True) for d in depths]).squeeze(1)
        states = torch.cat(states)

        leaves = batch_data(leaves, PAD, limit).contiguous()
        leaves = C(leaves, act(features), sizes) # either produce known vocab item or select (index of) word from source sentence

        leaves = leaves.view(leaves.shape[0] * leaves.shape[1], -1)
        Targ = Targ.contiguous().view(-1)

        gloss = ce(leaves, Targ)
        cl_loss = ce(sentiment, sentiment_target)
        # expected depth is the expected depth of a given branch of the syntax tree, multiplied by the length of the sentence
        depth_loss = mse(depths, expected_depth(sizes)) * 0.1
        # encourage average of states to be normally distributed
        swd = sliced_wasserstein_distance(states)

        loss = gloss + depth_loss + swd + cl_loss

        loss.backward()

        torch.nn.utils.clip_grad_norm_(chain(E.parameters(), G.parameters(), C.parameters(), CL.parameters(), charE.parameters()), 1.)

        optM.step()

        epoch_gloss += gloss.item()
        epoch_depth_loss += depth_loss.item()
        epoch_cl_loss += cl_loss.item()
        epoch_swd += swd.item()
        epoch_gacc += (leaves.log_softmax(-1).argmax(-1) == Targ).long().float().mean().item()
        epoch_cl_acc += (sentiment.softmax(-1).argmax(-1) == sentiment_target).long().float().mean().item()

    charE.eval()
    E.eval()
    G.eval()
    C.eval()
    CL.eval()

    tgloss = 0.
    tdepth_loss = 0.
    tswd = 0.
    tgacc = 0.
    tcl_loss = 0.
    tcl_acc = 0.

    test_words_data = ds.preprocess_new_observations('text', ds.test_df.text)
    test_chars_data = ds.preprocess_new_observations('chars', ds.test_df.text)
    test_sentiment_data = ds.preprocess_new_observations('airline_sentiment', ds.test_df.airline_sentiment)

    for ix, tb in enumerate(batch_indices(tixs, batch_size, tn_batches)): 

        tsentiment_target = torch.cat([test_sentiment_data['target'][i] for i in tb])
        test_words_targ = [
            torch.cat([w.unsqueeze(0) if w != 1 else torch.LongTensor([ix+model_config['io']]) for ix,w in enumerate(t)])
            for t in [test_words_data['vectors'][i] for i in tb]
        ]
        tchar_encodings = batch_data(
            [
                torch.cat([
                    charE(F.one_hot(w,num_classes=len(ds.vars['chars']['vocab']))\
                        .float().unsqueeze(0).transpose(1,2))
                    for w in s
                ])
                for s in [test_chars_data['vectors'][i] for i in tb]
            ],
            None,
            limit,
            use_pad_var=False
        )
        
        test_words = batch_data(list([test_words_data['vectors'][i] for i in tb]), PAD, limit)
        test_words_targ = batch_data(list(test_words_targ), PAD, limit)

        test_features, _ = E(test_words, tchar_encodings)

        test_trees = G(act(test_features.sum(0)), sizes=[limit] * len([test_words_data['sizes'][i] for i in tb]), return_trees=True)
        tleaves, tdepths, tstates = zip(*[
            (
                G.get_leaves(t),
                G.get_leaves(t,attr='depth'),
                G.get_states(t).unsqueeze(1).mean(0)
            )
            for t in test_trees
        ])

        tsentiment, _ = CL([G.get_states(t).unsqueeze(1) for t in test_trees], [G.get_states(t).unsqueeze(1) for t in test_trees])
        #act(E.embed(output_set)).unsqueeze(1))

        tdepths = torch.cat([d.sum(0,keepdim=True) for d in tdepths]).squeeze(1)
        tstates = torch.cat(tstates)

        tleaves = batch_data(tleaves, PAD, limit).contiguous()
        tleaves = C(tleaves, act(test_features), [test_words_data['sizes'][i] for i in tb])

        tleaves = tleaves.view(tleaves.shape[0] * tleaves.shape[1], -1)
        test_words_targ = test_words_targ.contiguous().view(-1)

        tgloss += ce(tleaves, test_words_targ).item()
        tdepth_loss += mse(tdepths, expected_depth([test_words_data['sizes'][i] for i in tb])).item() * 0.1
        tswd += sliced_wasserstein_distance(tstates).item()
        tcl_loss += ce(tsentiment, tsentiment_target).item()
        tgacc += (tleaves.log_softmax(-1).argmax(-1) == test_words_targ).long().float().mean().item()
        tcl_acc += (tsentiment.softmax(-1).argmax(-1) == tsentiment_target).long().float().mean().item()

    features, _ = E(T[:, :1], char_encodings[:, :1])
    trees = G(act(features.sum(0)), sizes=[limit], return_trees=True)
    leaves = batch_data([G.get_leaves(trees[0])], PAD, limit).contiguous()
    leaves = C(leaves, act(features), sizes[:1])

    tree = attach_to_leaves(trees[0], leaves, ds.vars['text'], model_config['io'], G, ds.vars['text']['text'][b[0]])

    print(f'Epoch: {e+1}')
    print(f'GLoss: {round(epoch_gloss / n_batches, 3)}')
    print(f'DepthLoss: {round(epoch_depth_loss / n_batches, 3)}')
    print(f'SWD: {round(epoch_swd / n_batches, 3)}')
    print(f'GAcc: {round(epoch_gacc / n_batches, 3)}')
    print(f'ClLoss: {round(epoch_cl_loss / n_batches, 3)}')
    print(f'ClAcc: {round(epoch_cl_acc / n_batches, 3)}')
    print(f'Test loss/depth/swd/acc/cl_acc: {", ".join(map(str, [round(tgloss / tn_batches, 3), round(tdepth_loss / tn_batches, 3), round(tswd / tn_batches, 3), round(tgacc / tn_batches, 3), round(tcl_acc / tn_batches, 3)]))}')
    print(f'Reconstruction of: {" ".join([w for w in ds.vars["text"]["text"][b[0]]])}')
    print(f'                   {" ".join([ds.vars["text"]["reverse_vocab"][w.item()] for w in T[:,0] if w.item() != 0])}')
    try:
        #inspect_parsed_sentence(ds.vars['text']['text'][b[0]], ds, E, G, C, 0, 'text', CL, output_set, sizes=sizes[:1])
        print_tree(tree, lambda x: x, attr='attachment')
    except Exception as e:
        print(e)
    print()
    torch.save(charE.state_dict(), f'{SAVE_DIR}/models/charE.pt')
    torch.save(E.state_dict(), f'{SAVE_DIR}/models/E.pt')
    torch.save(G.state_dict(), f'{SAVE_DIR}/models/G.pt')
    torch.save(C.state_dict(), f'{SAVE_DIR}/models/C.pt')
    torch.save(CL.state_dict(), f'{SAVE_DIR}/models/CL.pt')

