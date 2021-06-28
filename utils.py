import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.bernoulli import Bernoulli

from typing import Union, List
import numpy as np
import math
import random
from nltk.tree import Tree
import re
from collections import Counter
from itertools import chain
import pandas as pd
import pickle
import nltk

def get_n_branches(x):
    if type(x) is int:
        return x * 2 - 2
    return len(x) * 2 - 2

def expected_depth(x, mode='sum'):
    return torch.FloatTensor([
        (math.sqrt(x)) if mode != 'sum' else ((math.sqrt(x)) * (x)) # last sigmoid activation on a given branch should be < .5
    ])

def word_dropout(x, dropout=0.2, max_retries=5):
    if x.shape[0] < 3:
        dropout = 0.1
    retries = 0
    rw = Bernoulli(1. - dropout).sample(x.shape[:-1])

    while rw.sum().item() == 0. and retries < max_retries:
        rw = Bernoulli(1. - dropout).sample(x.shape[:-1])
        retries += 1
    if retries >= max_retries and rw.sum().item() == 0.:
        return x
    rw = rw.unsqueeze(2).repeat(1, 1, x.shape[-1])
    return rw * x
    
def span_dropout(x, dropout=0.15):
    if len(x) == 1:
        return x
    span_size = torch.poisson(torch.ones(1,1)).long().item() #random.choice(range(min(4, len(x))))
    if span_size == 0 or span_size > (len(x) - 1):
        return x
    mask_ix = random.choice(range(len(x)))
    return torch.cat([
        x[:mask_ix],
        torch.zeros((1, x.shape[-1])),
        x[mask_ix+span_size:],
    ]), (mask_ix, span_size)


def axe_recon_loss(leaves, target, p=4.0):
    return axe_loss(
        torch.cat([
            leaves.unsqueeze(0),
            torch.zeros((1, 1, leaves.shape[-1]))
        ], dim=1),
        torch.LongTensor((len(leaves),)),
        target.unsqueeze(0),
        torch.LongTensor((len(target),)),
        0,
        p
    )

def flatten(x):
    leaves = []
    def fxn(_x):
        if type(_x) is str:
            leaves.append(_x)
        else:
            fxn(_x[0])
            if len(_x) == 2:
                fxn(_x[1])
    fxn(x)
    return leaves

def vocab_encoder(V, s):
    return torch.LongTensor([0 if w not in V else V[w] for w in s])

def pad_to_match_length(x, y):
    if len(x) < len(y):
        diff = len(y) - len(x)
        if x.ndim == 2:
            return torch.cat([x, torch.zeros((diff, x.shape[-1]))]), y
        else:
            return torch.cat([x, torch.zeros((diff,)).long()]), y
    elif len(x) > len(y):
        diff = len(x) - len(y)
        return x, torch.cat([y, torch.zeros((diff,)).long()])
    return x, y

def create_target(x, o):
    return torch.cat([
        w.unsqueeze(0) if w != 1 else torch.LongTensor([ix+o])
        for ix,w in enumerate(x)
    ])

def accuracy(x, y):
    return (x.softmax(-1).argmax(-1) == y).long().float().mean().item()

def print_tree(x, transform=lambda x: x, attr='terminal'):
    nx = [None]
    def fx(x, nx):
        if x['left'] == {}:
            if attr is not None:
                nx[0] = transform(x[attr])
            else:
                nx[0] = transform(x)
        else:
            nx[0] = [None]
            nx += [[None]]
            fx(x['left'], nx[0])
            fx(x['right'], nx[1])
    fx(x, nx)
    nx = Tree.fromstring(str(nx).replace('(','{').replace(')','}').replace('[','(').replace(']',')').replace('),',')'))
    nx.pretty_print()

def attach_to_leaves(tree, leaves, V, io, G, source):
    leaves = [
        w if w not in V else V[w].replace('[','{').replace(']','}')
        for w in leaves.squeeze(1).softmax(-1).argmax(-1).tolist()
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

def parse_sentence(x, ds, P, model_config, add_end_mask=False, _print_tree=True, return_encoding=False):
    _d, c, d = ds.preprocess(pd.DataFrame.from_dict({'text': [x]}), ['text'], flatten=True)['text'][0]
    parse_dict = P(_d[0], c[0])#, add_end_mask=add_end_mask)
    if return_encoding:
        return parse_dict['encoding']
    tree = attach_to_leaves(parse_dict['tree'], parse_dict['leaves_after_copy'], ds.vrs['text']['rV'], model_config['i'], P.decoder, d[0])
    if _print_tree:
        print_tree(tree, lambda x: x, attr='attachment')
    return tree

def get_characters(tree, leaves, ds, model_config, G, source):
    tree = attach_to_leaves(tree, leaves, ds.vrs['text']['rV'], model_config['i'], G, source)
    attachment = G.get_leaves(tree, cat=False, attr='attachment')
    return leaves, attachment, [torch.cat(c, dim=-1) for c in ds.character_encoder(ds.vrs['text']['cV'], lambda x: x, attachment)]

def preprocessor(x, lower=True):
    def remove_extra_space(x):
        return re.sub(r'\s{2,}', r' ', x)
    x = re.sub(r'([A-Z][a-z]{,2})\. ([A-Z])', r'\1 \2', x) # capture titles such as Dr.
    x = re.sub(r'([bd])\.', r'\1 ', x) # capture birth and death dates from wiki articles
    x = re.sub(r'(\[ \d+ \]|\[\d+\])', r'', x)
    x = re.sub(r'(Late|Cancel)(\w*) Flight(\w*)', r'\1\3', x)
    if lower:
        x = x.lower()
        
    x = re.sub(r'&amp;', 'and', x)
    x = re.sub(r'&gt;', '>', x)
    x = re.sub(r'&lt;', '<', x)
    x = re.sub(r'(http\S+)',r'<u>',x)
    x = re.sub(r'(\D)\1{2,}', r'\1', x)
    x = re.sub(r'(\W)', r' \1 ', x)
    x = re.sub(r'(\d) (\.|-|/) (\d)', r'\1\2\3', x)
    x = re.sub(r'(#|@) (\S)', r'\1\2', x)
    x = re.sub(r'< u >', r'<u>', x)
    x = remove_extra_space(x)
    x = re.sub(r'(\?) (!)', r'\1\2', x)
    x = re.sub(r'(!) (\?)', r'\1\2', x)
    x = re.sub(r'(!\?|\?!)+', r'?', x)
    x = re.sub(r'\.{2,}', r'.', x)
    return [remove_extra_space(s.strip()).split(' ') for s in nltk.sent_tokenize(x) if s.strip().split(' ') != [' '] and 'mw - parse' not in s]
    
def word_reconstruction_loss(parse_dict, target):
    leaves = parse_dict['leaves_after_copy']
    leaves, target = pad_to_match_length(leaves, target)
    return axe_recon_loss(leaves, target, 2.) 
    
def word_depth_loss(parse_dict, target, mse=nn.MSELoss(), coeff=0.001):
    depth = parse_dict['depths']
    return mse(depth.sum(0), expected_depth(len(target), mode='sum')) * coeff
    
def word_swd(parse_dict):
    states = parse_dict['states']
    return sliced_wasserstein_distance(states)
    
def word_accuracy(parse_dict, target):
    leaves= parse_dict['leaves_after_copy']
    leaves, target = pad_to_match_length(leaves, target)
    return accuracy(leaves.detach(), target)
    
def char_reconstruction_loss(parse_dict, target):
    char_trees = parse_dict['char_trees']
    return torch.cat([
        axe_recon_loss(
            pad_to_match_length(c['leaves'], t.transpose(1,2).squeeze(0).argmax(-1))[0],
            t.transpose(1,2).squeeze(0).argmax(-1),
            2.
        ).view(1, 1)
        for c,t in zip(char_trees, target)
    ]).mean(0, keepdim=True)
    
def char_depth_loss(parse_dict, target, mse=nn.MSELoss(), coeff=0.001):
    char_trees = parse_dict['char_trees']
    return torch.cat([
        mse(
            c['depths'].sum(0),
            expected_depth(t.shape[-1], mode='sum')
        ).view(1,1) * coeff
        for c,t in zip(char_trees, target)
    ]).mean(0, keepdim=True)
    
def char_swd(parse_dict):
    char_trees = parse_dict['char_trees']
    return torch.cat([
        sliced_wasserstein_distance(c['states']).view(1,1)
        for c in char_trees
    ]).mean(0, keepdim=True)
    
def char_accuracy(parse_dict, target):
    char_trees = parse_dict['char_trees']
    acc = [
        accuracy(
            pad_to_match_length(c['leaves'], t.transpose(1,2).squeeze(0).argmax(-1))[0].detach(),
            t.transpose(1,2).squeeze(0).argmax(-1)
        )
        for c,t in zip(char_trees, target)
    ]
    return sum(acc)/len(acc)
    
def consistency_loss(_d, c, P, mse=nn.MSELoss(), coeff=1.):
    states0 = (lambda x: P.decoder.get_leaves(x, 'depth'))(P(_d[0], c[0])['tree'])
    states1 = (lambda x: P.decoder.get_leaves(x, 'depth'))(P(_d[0], c[0])['tree'])        
    if len(states0) < len(states1):
        states0 = torch.cat([states0, torch.zeros((len(states1) - len(states0), states0.shape[-1]))])
    elif len(states1) < len(states0):
        states1 = torch.cat([states1, torch.zeros((len(states0) - len(states1), states1.shape[-1]))])
    return mse(states0, states1) * coeff
    
def unsupervised_clustering_loss(data, ds, P, model_config, kmeans, sample_size=30, n_shots=3, ce=nn.CrossEntropyLoss()):
    random_sample = data.sample(sample_size).tolist()
    encodings = [
        st['state']
        for s in random_sample
        for st in P.decoder.get_subtrees(parse_sentence(s, ds, P, model_config, _print_tree=False), len(preprocessor(s)[0]))
    ]
    cluster_ids = kmeans.fit_predict(torch.cat(encodings).detach()).tolist()
    grouped_encodings = {i:[] for i in set(cluster_ids)}
    for encoding,i in zip(encodings, cluster_ids):
        grouped_encodings[i] += [encoding]
    grouped_encodings = {i:random.sample(es, n_shots) for i,es in grouped_encodings.items() if len(es) >= n_shots}
    grouped_encodings = {i:es for i,es in enumerate(grouped_encodings.values())}
    supports = [torch.cat(es[:-1]).mean(0, keepdim=True) for es in grouped_encodings.values()]
    queries = [es[-1] for es in grouped_encodings.values()]
    dists = []
    outputs = []
    for ix,(support, query) in enumerate(zip(supports, queries)):
        p_dist = -((support - query).pow(2).sum()).view(1,1)
        dists = [
            -((negatives - query).pow(2).sum()).view(1,1)
            for negatives in [s for i,s in enumerate(supports) if ix != i]
        ]
        dists.insert(ix, p_dist)
        outputs += [[torch.cat(dists, dim=1), torch.LongTensor([ix])]]
    x, y = zip(*outputs)
    acc = (torch.cat(x).softmax(-1).argmax(-1) == torch.cat(y)).long().float().mean().item()
    return ce(torch.cat(x), torch.cat(y)), acc  

class Dataset:
    def __init__(self, config, saved_variables=None):
        if saved_variables is not None:
            self.load_variables(saved_variables)
        else:
            self.df = config['df']
            self.vrs = config['vrs']
            self.limit = config['limit']
            for k in self.vrs:
                self.create_variable(k)
    def load_variables(self, vrs):
        self.vrs = { k: {} for k in vrs.keys() if k not in ['limit', 'df'] }
        self.limit = vrs['limit']
        for v in vrs.keys():
            if v in ['limit','df']:
                continue
            self.vrs[v]['preprocessor'] = vrs[v]['preprocessor']
            self.vrs[v]['V'] = vrs[v]['V']
            self.vrs[v]['cV'] = vrs[v]['cV']
            self.vrs[v]['rV'] = vrs[v]['rV']
    def create_variable(self, v):
        name = v
        v = self.vrs[v]
        preprocessor = v['preprocessor']
        vocab_min_freq= v['vocab_min_freq']
        char_min_freq = v['char_min_freq']
        V = v['base_vocab']
        lV = len(list(V.keys()))
        if v['rank'] is None:
            _V = {
                w:(ix+lV)
                for ix,w in enumerate([
                    k for k,f in Counter(chain(*[s for s in chain(*self.df[v['reference']].map(preprocessor)) if len(s) <= self.limit])).items() if (f >= vocab_min_freq and k.islower()) # k.lower() avoids duplicate words that differ only the use of uppercase letters and discourages proper nouns
                ])
            }
        else:
            _V = list(reversed(sorted(
                Counter(chain(*[s for s in chain(*self.df[v['reference']].map(preprocessor)) if len(s) <= self.limit])).items(),
                key=lambda x: x[1]
            )))
            cutoff = round(len(_V) * v['rank'])
            _V = {w:(ix+lV) for ix,(w,_) in enumerate(_V[:cutoff])}
        V.update(_V)
        cV = { '<unk>': 0 }
        cV.update({
            c:(ix+1)
            for ix,c in enumerate([
                k for k,f in Counter(chain(*chain(*[s for s in chain(*self.df[v['reference']].map(preprocessor)) if len(s) <= self.limit]))).items() if f >= char_min_freq
            ])
        })
        rV = {ix:w for w,ix in V.items()}
        self.vrs[name]['V'], self.vrs[name]['cV'], self.vrs[name]['rV'], self.vrs[name]['reference'] = V, cV, rV, v['reference']
    

    def preprocess(self, data, vs, flatten=False, embedder=None, use_reference=False):
        if len(vs) > 1:
            flatten = False
        output = {}
        for ix, v in enumerate(vs):
            name = v
            v = self.vrs[v]
            reference = v['reference'] if use_reference else name
            words = [ self.word_encoder(v['V'], v['preprocessor'], s, embedder=(None if embedder is None else embedder[1] if embedder[0] == name else None)) for s in data.iloc[:,ix] ]
            characters = [ self.character_encoder(v['cV'], v['preprocessor'], s) for s in data.iloc[:,ix] ]
            if flatten:
                words = map(lambda x: [x], chain(*words))
                characters = map(lambda x: [x], chain(*characters))
                data = map(lambda x: [x], chain(*map(v['preprocessor'], data.iloc[:,ix])))
                output[reference] = [o for o in zip(words, characters, data) if len(o[0][0]) <= self.limit]
            else:
                # following lines require that length limiting variable must be first in list of vrs
                if ix == 0:
                    tmp = [(ii,o) for ii,o in enumerate(zip(words, characters, data.iloc[:,ix])) if len(o[0][0]) <= self.limit]
                    ixs, _data = zip(*tmp)
                    output[reference] = _data
                else:
                    output[reference] = [o for ii,o in enumerate(zip(words, characters, data.iloc[:,ix])) if ii in ixs]
        return output
        
    def word_encoder(self, V, p, s, embedder):
        if embedder is not None:
            return [
                torch.cat([embedder(w) for w in _s])
                for _s in p(s)
            ]
        return [
            torch.LongTensor([1 if w not in V else V[w] for w in _s])
            for _s in p(s)
        ]

    def character_encoder(self, V, p, s):
        return [
            [
                F.one_hot(
                    torch.LongTensor([0 if c not in V else V[c] for c in w]),
                    num_classes=len(V)
                ).float().unsqueeze(0).transpose(1,2)
                for w in _s
            ]
            for _s in p(s)
        ]
        
def save_parser(parser, optimizer, var, model_config, df, path):
    df.to_pickle(path + '/df.pkl')
    with open(path + '/V.pkl', 'wb') as f:
        pickle.dump(var['V'], f)
    with open(path + '/cV.pkl', 'wb') as f:
        pickle.dump(var['cV'], f)
    with open(path + '/rV.pkl', 'wb') as f:
        pickle.dump(var['rV'], f)
    with open(path + '/model_config.pkl', 'wb') as f:
        pickle.dump(model_config, f)
    torch.save(parser.state_dict(), path + '/model.pt')
    torch.save(parser.state_dict(), path + '/optimizer.pt')


# other people's code below

def axe_loss(logits: torch.FloatTensor,
             logit_lengths: torch.Tensor,
             targets: torch.LongTensor,
             target_lengths: torch.Tensor,
             blank_index: torch.LongTensor,
             delta: torch.FloatTensor,
             reduction: str = 'mean',
             label_smoothing: float = None,
             return_a: bool = False
            ) -> Union[torch.FloatTensor, List[torch.Tensor]]:
    """Aligned Cross Entropy
    Marjan Ghazvininejad, Vladimir Karpukhin, Luke Zettlemoyer, Omer Levy, in arXiv 2020
    https://arxiv.org/abs/2004.01655
    Computes the aligned cross entropy loss with parallel scheme.
    Parameters
    ----------
    logits : ``torch.FloatTensor``, required.
        A ``torch.FloatTensor`` of size (batch_size, sequence_length, num_classes)
        which contains the unnormalized probability for each class.
    logit_lengths : ``torch.Tensor``, required.
        A ``torch.Tensor`` of size (batch_size,)
        which contains lengths of the logits
    targets : ``torch.LongTensor``, required.
        A ``torch.LongTensor`` of size (batch, sequence_length) which contains the
        index of the true class for each corresponding step.
    target_lengths : ``torch.Tensor``, required.
        A ``torch.Tensor`` of size (batch_size,)
        which contains lengths of the targets
    blank_index : ``torch.LongTensor``, required.
        A ``torch.LongTensor``, An index of special blank token.
    delta : ``torch.FloatTensor``, required.
        A ``torch.FloatTensor`` for penalizing skip target operators.
    reduction : ``str``, optional.
        Specifies the reduction to apply to the output.
        Default "mean".
    label_smoothing : ``float``, optional
        Whether or not to apply label smoothing.
    return_a : ``bool``, optional.
        Whether to return the matrix of conditional axe values. Default is False.
    """
    
    assert targets.size(0) == logits.size(0), f'Inconsistency of batch size,  {targets.size(0)} of targets and {logits.size(0)} of logits.'

    batch_size, logits_sequence_length, num_class = logits.shape
    _, target_sequence_length = targets.shape

    device = logits.device
    
    # for torch.gather
    targets = targets.unsqueeze(-1) # batch_size, target_sequence_length, 1

    # (batch_size, target_sequence_length + 1, logits_sequence_length + 1)
    batch_A = torch.zeros(targets.size(0), targets.size(1) + 1, logits.size(1) + 1).to(device)
    batch_blank_index = torch.full((logits.size(0), 1), blank_index, dtype = torch.long).to(device)

    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    
    # A_{i,0} = A_{i−1,0} − delta * log P_1 (Y_i)
    for i in range(1, targets.size(1) + 1):
        # batch_A[:, i, 0] is calculated from targets[:, i-1, :], because batch_A added 0-th row
        batch_A[:, i, 0] = batch_A[:, i-1, 0] - delta * torch.gather(log_probs[:, 0, :], dim=1, index=targets[:, i-1, :]).squeeze(-1)

    # A_{0,j} = A_{0,j−1} − log P_j ("BLANK")
    for j in range(1, logits.size(1) + 1):
        # batch_A[:, 0, j] is calculated from log_probs[:, j-1, :], because batch_A added 0-th column
        batch_A[:, 0, j] = batch_A[:, 0, j-1] - delta * torch.gather(log_probs[:, j-1, :], dim=1, index=batch_blank_index).squeeze(-1)


    # flip logit dim to get anti-diagonal part by using use torch.diag
    batch_A_flip = batch_A.flip(-1) # (batch_size, target_sequence_length + 1, logits_sequence_length + 1)
    log_probs_flip = log_probs.flip(-2) # (batch_size, sequence_length, num_classes)

    # to extract indices for the regions corresponding diag part.
    map_logits = torch.arange(logits.size(1)) - torch.zeros(targets.size(1), 1)
    map_targets = torch.arange(targets.size(1)).unsqueeze(-1) - torch.zeros((1, logits.size(1)))
    # index must be int
    map_logits = map_logits.long().to(device)
    map_targets = map_targets.long().to(device)

    for i in range(logits.size(1) - 1, -targets.size(1), -1):
        
        
        # batch_A_flip_sets[:, :, :, 0] : batch_A[:, i  , j-1]
        # batch_A_flip_sets[:, :, :, 1] : batch_A[:, i-1, j  ]
        # batch_A_flip_sets[:, :, :, 2] : batch_A[:, i-1, j-1]
        batch_A_flip_sets = torch.cat((batch_A_flip.roll(shifts=-1, dims=-1).unsqueeze(-1),
                                       batch_A_flip.roll(shifts= 1, dims=-2).unsqueeze(-1),
                                       batch_A_flip.roll(shifts=(1, -1), dims=(-2, -1)).unsqueeze(-1)),
                                       dim = -1)
        
        # trimming
        # - the last column (A_{0,j} = A_{0,j−1} − log P_j ("BLANK")) 
        # - the first row (A_{i,0} = A_{i−1,0} − delta * log P_1 (Y_i))
        batch_A_flip_sets_trim = batch_A_flip_sets[:, 1:, :-1, :]

        # extracting anti-diagonal part
        # (batch, 3, num_diag)
        A_diag = batch_A_flip_sets_trim.diagonal(offset=i, dim1 = -3, dim2 = -2)
        
        # (batch, num_diag, 3)
        A_diag = A_diag.transpose(-1, -2)
        num_diag = A_diag.size(1)
        
        logit_indices = map_logits.diagonal(offset=i, dim1 = -2, dim2 = -1)
        # log_probs_diag : (batch, num_diag, num_class)
        log_probs_flip_diag = log_probs_flip[:, logit_indices[0]:logit_indices[-1]+1, :]

        target_indices = map_targets.diagonal(offset=i, dim1 = -2, dim2 = -1)
        # targets_diag : (batch, num_diag, num_class)
        targets_diag = targets[:, target_indices[0]:target_indices[-1]+1, :]

        # align, skip_prediction, skip_target
        batch_align = A_diag[:, :, 2] - torch.gather(log_probs_flip_diag, dim=2, index=targets_diag).squeeze(-1)
        batch_skip_prediction = A_diag[:, :, 0] - torch.gather(log_probs_flip_diag, dim=2, index=batch_blank_index.expand(-1, num_diag).unsqueeze(-1)).squeeze(-1)
        batch_skip_target = A_diag[:, :, 1] - delta * torch.gather(log_probs_flip_diag, dim=2, index=targets_diag).squeeze(-1)

        # (batch_size, num_diag, 3)
        operations = torch.cat((batch_align.unsqueeze(-1), batch_skip_prediction.unsqueeze(-1), batch_skip_target.unsqueeze(-1)), dim = -1)

        # (batch_size, num_diag)
        diag_axe = torch.min(operations, dim = -1).values
        
        assert logits.size(1) > targets.size(1), "assuming target length < logit length." 

        if i > (logits.size(1) - targets.size(1)):
            # (batch_size, logits_length, logits_length)
            # -> (batch_size, targets_length, logits_length)
            axe = torch.diag_embed(diag_axe, offset=i, dim1=-2, dim2=-1)
            batch_A_flip[:, 1:, :-1] += axe[:, :targets.size(1), :]
        elif i > 0:
            # (batch_size, logits_length, logits_length)
            # -> (batch_size, targets_length, logits_length)
            axe = torch.diag_embed(diag_axe, offset=0, dim1=-2, dim2=-1)
            batch_A_flip[:, 1:, i : i + targets.size(1)] += axe
        else:
            axe = torch.diag_embed(diag_axe, offset=i, dim1=-2, dim2=-1)
            batch_A_flip[:, 1:, :targets.size(1)] += axe

    # recover correct order in logit dim
    batch_A = batch_A_flip.flip(-1)
    
    # rm 0-th row and column
    _batch_A = batch_A[:, 1:, 1:]

    ## Gather A_nm, avoiding masks
    # index_m : (batch_size, target_sequence_length, 1)
    index_m = logit_lengths.unsqueeze(-1).expand(-1, _batch_A.size(1)).unsqueeze(-1).long()

    # gather m-th colmun
    # batch_A_nm : (batch_size, target_sequence_length, 1)
    # index_m occors out of bounds for index 
    batch_A_m = torch.gather(_batch_A, dim=2, index=(index_m - 1))
    batch_A_m = batch_A_m.squeeze(-1)

    # index_n : (batch_size, 1)
    index_n = target_lengths.unsqueeze(-1).long()
    
    # gather n-th row
    # batch_A_nm : (batch_size, 1, 1)
    batch_A_nm = torch.gather(batch_A_m, dim=1, index=(index_n - 1))

    # batch_A_nm : (batch_size)
    batch_A_nm = batch_A_nm.squeeze(-1)

    if reduction == "mean":
        axe_nm = batch_A_nm.mean()
    else:
        raise NotImplementedError
    
    # Refs fairseq nat_loss.
    # https://github.com/pytorch/fairseq/blob/6f6461b81ac457b381669ebc8ea2d80ea798e53a/fairseq/criterions/nat_loss.py#L70
    # actuary i'm not sure this is reasonable.
    if label_smoothing is not None and label_smoothing > 0.0:
        axe_nm = axe_nm * (1.0-label_smoothing) - log_probs.mean() * label_smoothing

    if return_a:
        return axe_nm, batch_A.detach()
    
    return axe_nm

def rand_projections(embedding_dim, num_samples=50, SD=1.):
    """This function generates `num_samples` random samples from the latent space's unit sphere.
        Args:
            embedding_dim (int): embedding dimensionality
            num_samples (int): number of random projection samples
        Return:
            torch.Tensor: tensor of size (num_samples, embedding_dim)
    """
    projections = [w / np.sqrt((w**2).sum())  # L2 normalization
                   for w in np.random.normal(0., SD, (num_samples, embedding_dim))]#size=(num_samples, embedding_dim))]
    projections = np.asarray(projections)
    return torch.from_numpy(projections).type(torch.FloatTensor)


def _sliced_wasserstein_distance(encoded_samples,
                                 distribution_samples,
                                 num_projections=50,
                                 p=2):
    """ Sliced Wasserstein Distance between encoded samples and drawn distribution samples.
        Args:
            encoded_samples (toch.Tensor): tensor of encoded training samples
            distribution_samples (torch.Tensor): tensor of drawn distribution training samples
            num_projections (int): number of projections to approximate sliced wasserstein distance
            p (int): power of distance metric
        Return:
            torch.Tensor: tensor of wasserstrain distances of size (num_projections, 1)
    """
    # derive latent space dimension size from random samples drawn from latent prior distribution
    embedding_dim = distribution_samples.size(1)
    # generate random projections in latent space
    projections = rand_projections(embedding_dim, num_projections)
    # calculate projections through the encoded samples
    encoded_projections = encoded_samples.matmul(projections.transpose(0, 1))
    # calculate projections through the prior distribution random samples
    distribution_projections = (distribution_samples.matmul(projections.transpose(0, 1)))
    # calculate the sliced wasserstein distance by
    # sorting the samples per random projection and
    # calculating the difference between the
    # encoded samples and drawn random samples
    # per random projection
    wasserstein_distance = (torch.sort(encoded_projections.transpose(0, 1), dim=1)[0] -
                            torch.sort(distribution_projections.transpose(0, 1), dim=1)[0])
    # distance between latent space prior and encoded distributions
    # power of 2 by default for Wasserstein-2
    wasserstein_distance = torch.pow(wasserstein_distance, p)
    # approximate mean wasserstein_distance for each projection
    return wasserstein_distance.mean()


def sliced_wasserstein_distance(encoded_samples,
                                num_projections=50,
                                p=2,
                                device='cpu',
                                SD=1.):
    """ Sliced Wasserstein Distance between encoded samples and drawn distribution samples.
        Args:
            encoded_samples (toch.Tensor): tensor of encoded training samples
            distribution_samples (torch.Tensor): tensor of drawn distribution training samples
            num_projections (int): number of projections to approximate sliced wasserstein distance
            p (int): power of distance metric
            device (torch.device): torch device (default 'cpu')
        Return:
            torch.Tensor: tensor of wasserstrain distances of size (num_projections, 1)
    """
    # draw random samples from latent space prior distribution
    z = torch.normal(0., SD, encoded_samples.shape) #randn(encoded_samples.shape)
    # approximate mean wasserstein_distance between encoded and prior distributions
    # for each random projection
    swd = _sliced_wasserstein_distance(encoded_samples, z,
                                       num_projections, p)
    return swd
