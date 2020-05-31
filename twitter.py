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

from utils import *
from models import *

r000 = lambda x: re.sub('&amp;', 'and', x.lower())
r00 = lambda x: re.sub(r'(…|\.\.\.)', r'.', x)
r0 = lambda x: re.sub('’', r"'", x)
r1 = lambda x: re.sub('(”|“)', r'"', x) 
r2 = lambda x: re.sub(r'(http\S+)',r'<u>',x)
r3 = lambda x: re.sub(r'(.)\1{2,}', r'\1', x) # must run before r4; captures any character that repeats more than twice ('iiii', '!!!')
r4 = lambda x: re.sub(r'([^a-z])\1+', r'\1', x) # captures any non alphebetical character that repeats at all
r5 = lambda x: re.sub(r'([a-z])([^a-z])([a-z])', r'\1 \2 \3', x.lower()) #re.sub(r'(\S)([.!?’"\',$/:\(\)])(\S)'  
r6 = lambda x: re.sub(r'([a-z])([^a-z])', r'\1 \2', x.lower()) #re.sub(r'(\S)([.!?’"\',$/:\(\)])', r'\1 \2', x.lower())
r7 = lambda x: re.sub(r'([^a-z])([a-z])', r'\1 \2', x.lower()) #re.sub(r'([.!?’"\',$/:\(\)])(\S)', r'\1 \2', x.lower())
r8 = lambda x: x.replace('\n',' ')
#r9 = lambda x: x.replace('&amp;','and')
r10 = lambda x: re.sub(r'(@) (\S+)', r'\1', x)
r11 = lambda x: re.sub(r'(#) (\S+)', r'\1', x)
r12 = lambda x: re.sub(r'((\$?[0-9]+[.,:/ -]?[0-9]+)+|[0-9])', r'<n>', x)
r13 = lambda x: re.sub(r'\s{2,}',r' ', x)
r14 = lambda x: re.sub(r'(^@ | (<u>|#)$)', r'', x)
r15 = lambda x: re.sub(r'(\S)(<n>)', r'\1 \2', x)
r16 = lambda x: re.sub(r'< (u|n) >', r'<\1>', x)
r17 = lambda x: re.sub(r'(<n>)(\S)', r'\1 \2', x)

preprocessor = lambda x: [w for w in r13(r17(r16(r15(r14(r13(r12(r11(r10(r8(r7(r6(r5(r4(r3(r2(r1(r0(r00(r000(x)))))))))))))))))))).strip().split(' ')]

limit = 10
config = {
    'limit': limit,
    'df': pd.read_csv('./data/Tweets.csv'),
    'anchor': 'text',
    'sample_size': 2052,
    'test_size': 128,
    'unify': ['text','airline_sentiment'],
    'sleep': 0.5,
    'reserve': { 'text' : ['negative','positive','neutral'] },
    'vars': {
        'text': {
            'preprocessor': preprocessor,
            'limit': 2,
            'pad': True,
            'extra_fxns': {
                'sizes': ('text', lambda context, x: len(x))
             },
        },
        'chars': {
            'source': 'text',
            'preprocessor': lambda x: list(map(list, preprocessor(x))),
            'limit': 1,
            'pad': True,
            'extra_fxns': {
                'sizes': ('text', lambda context, x: list(map(len, x)))
            }
        },
        'airline_sentiment': {
            'preprocessor': lambda x: [x],
            'limit': 0,
            'extra_fxns': {
                'target': ('vectors', lambda context, x: torch.LongTensor([len(context['vocab']) - (x+1)]))
            }
        }
    }
}

ds = Dataset(config)

output_set = torch.LongTensor([
    ds.vars['text']['vocab']['positive'],
    ds.vars['text']['vocab']['neutral'],
    ds.vars['text']['vocab']['negative']
])

PAD = 0
SD = 1.

train_sample_size = ds.sample_size - config['test_size']

si = partial(shuffle_indices, list(range(train_sample_size))) # returns shuffled list of indices
tsi = partial(shuffle_indices, list(range(config['test_size'])))

batch_size = 64
n_epochs = 200
n_batches = train_sample_size // batch_size
tn_batches = config['test_size'] // batch_size
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
    'wd': .05,
    'pad': PAD,
    'sentiment': len(ds.vars['airline_sentiment']['vocab']),
    'limit': limit,
    'dr': None,
    #'has_subsequence': True,
}

char_model_config = deepcopy(model_config)
char_model_config.update({ 'wd': None, 'h': 64, 'adaptor': h, 'has_subsequence': False, 'io': len(ds.vars['chars']['vocab']) })
classifier_config = deepcopy(model_config)
classifier_config.update({ 'io': 3, 'n_heads': 2 })
copy_config = deepcopy(model_config)
copy_config.update({ 'n_heads': None })

charE = Encoder(**char_model_config)
E = Encoder(**model_config)
G = Generator(**model_config)
C = Copy(**copy_config)
CL = Classifier(**classifier_config)

charE.apply(weight_init)
E.apply(weight_init)
G.apply(weight_init)
C.apply(weight_init)
CL.apply(weight_init)

C.V.weight.data = E.embed.weight

optM = optim.AdamW(chain(E.parameters(), G.parameters(), C.parameters(), CL.parameters()), amsgrad=True)

for e in range(n_epochs):

    ixs = si()
    tixs = tsi()
    epoch_gloss = 0.
    epoch_gacc = 0.
    epoch_depth_loss = 0.
    epoch_swd = 0.
    epoch_cl_loss = 0.
    epoch_cl_acc = 0.

    E.train()
    G.train()
    charE.train()
    C.train()
    CL.train()

    for ix, b in enumerate(batch_indices(ixs, batch_size, n_batches)): 

        T, Tchars, sizes, sentiment_target = zip(*[
            (
                ds.vars['text']['vectors'][i],
                ds.vars['chars']['vectors'][i],
                ds.vars['text']['sizes'][i],
                ds.vars['airline_sentiment']['target'][i]
            )
            for i in b
        ])

        # zero tensors added to end ensure all observations are the same length
        Targ = [torch.cat([w.unsqueeze(0) if w != 1 else torch.LongTensor([ix+model_config['io']]) for ix,w in enumerate(t)]) for t in T]
        sentiment_target = torch.cat(sentiment_target)

        _Tchars = batch_data([torch.cat([charE(w.unsqueeze(1)) for w in s]) for s in Tchars], PAD, limit, h)
        T = batch_data(list(T), PAD, limit)
        Targ = batch_data(list(Targ), PAD, limit)

        optM.zero_grad()

        features = E(T)#, _Tchars)

        trees = G(act(features.sum(0)), sizes=[limit] * len(sizes), return_trees=True) #sizes

        leaves, depths, states = zip(*[
            (
                G.get_leaves(t),
                G.get_leaves(t,attr='depth'), 
                G.get_states(t).unsqueeze(1).mean(0)
            )
            for t in trees
        ])

        sentiment, _ = CL([G.get_states(t).unsqueeze(1) for t in trees], act(E.embed(output_set)).unsqueeze(1))

        depths = torch.cat([d.sum(0,keepdim=True) for d in depths]).squeeze(1)
        states = torch.cat(states)

        leaves, Tg = pad(define_padded_vectors(nn.utils.rnn.pad_sequence(leaves), PAD), Targ)
        leaves = C(leaves, act(features), sizes)
        
        leaves = leaves.view(leaves.shape[0] * leaves.shape[1], -1)
        Tg = Tg.contiguous().view(-1)

        gloss = ce(leaves, Tg)
        cl_loss = ce(sentiment, sentiment_target)
        depth_loss = mse(depths, expected_depth(sizes))
        swd = sliced_wasserstein_distance(states)

        loss = gloss + depth_loss + swd + cl_loss

        loss.backward()

        optM.step()

        epoch_gloss += gloss.item()
        epoch_depth_loss += depth_loss.item()
        epoch_cl_loss += cl_loss.item()
        epoch_swd += swd.item()
        epoch_gacc += (leaves.log_softmax(-1).argmax(-1) == Tg).long().float().mean().item()
        epoch_cl_acc += (sentiment.softmax(-1).argmax(-1) == sentiment_target).long().float().mean().item()

    E.eval()
    G.eval()
    charE.eval()
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
        
        test_chars = batch_data([torch.cat([charE(w.unsqueeze(1)) for w in s]) for s in [test_chars_data['vectors'][i] for i in tb]], PAD, limit, h)
        test_words = batch_data(list([test_words_data['vectors'][i] for i in tb]), PAD, limit)
        test_words_targ = batch_data(list(test_words_targ), PAD, limit)

        test_features = E(test_words)#, test_chars)

        test_trees = G(act(test_features.sum(0)), sizes=[limit] * len([test_words_data['sizes'][i] for i in tb]), return_trees=True)
        # [test_words_data['sizes'][i] for i in tb]
        tleaves, tdepths, tstates = zip(*[
            (
                G.get_leaves(t),
                G.get_leaves(t,attr='depth'),
                G.get_states(t).unsqueeze(1).mean(0)
            )
            for t in test_trees
        ])

        tsentiment, _ = CL([G.get_states(t).unsqueeze(1) for t in test_trees], act(E.embed(output_set)).unsqueeze(1))

        tdepths = torch.cat([d.sum(0,keepdim=True) for d in tdepths]).squeeze(1)
        tstates = torch.cat(tstates)

        tleaves, test_words_targ = pad(define_padded_vectors(nn.utils.rnn.pad_sequence(tleaves), PAD), test_words_targ)
        tleaves = C(tleaves, act(test_features), [test_words_data['sizes'][i] for i in tb]) #[limit]*batch_size)

        tleaves = tleaves.view(tleaves.shape[0] * tleaves.shape[1], -1)
        test_words_targ = test_words_targ.contiguous().view(-1)

        tgloss += ce(tleaves, test_words_targ).item()
        tdepth_loss += mse(tdepths, expected_depth([test_words_data['sizes'][i] for i in tb])).item()
        tswd += sliced_wasserstein_distance(tstates).item()
        tcl_loss += ce(tsentiment, tsentiment_target).item()
        tgacc += (tleaves.log_softmax(-1).argmax(-1) == test_words_targ).long().float().mean().item()
        tcl_acc += (tsentiment.softmax(-1).argmax(-1) == tsentiment_target).long().float().mean().item()


    features = E(T[:, :1])#, _Tchars[:, :1])
    trees = G(act(features.sum(0)), sizes=[limit], return_trees=True) # [limit]
    leaves, _ = pad(G.get_leaves(trees[0]).unsqueeze(1), T[:,:1])
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
        inspect_parsed_sentence(ds.vars['text']['text'][b[0]], ds, E, G, C, charE, 0, 'text', None,
             CL, output_set, None, sizes=sizes[:1])
        #print_tree(ds.vars['chars']['text'][b[0]],
        #    tree,
        #    lambda x: x,
        #    attr='attachment'
        #)
    except Exception as e:
        print(e)
    print()
