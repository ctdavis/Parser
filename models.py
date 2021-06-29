import torch
import torch.nn as nn
import torch.nn.functional as F
import torchcontrib.nn as tnn

import random

from utils import *

wn = torch.nn.utils.weight_norm

class CharacterEncoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super(CharacterEncoder, self).__init__()
        i, h = kwargs['char_i'], kwargs['h']
        self.selu = nn.SELU()
        self.conv0 = wn(nn.Conv1d(i, h, 3, 1, 1))
        self.conv1 = wn(nn.Conv1d(h, h, 3, 1, 1))
    def forward(self, x):
        c0 = self.selu(self.conv0(x))
        c1 = self.conv1(c0).sum(-1)
        return c1
        
class CNNEncoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super(CNNEncoder, self).__init__()
        self.selu = nn.SELU()
        self.char_encoder = CNNEncoder_(**kwargs)
        self.word_encoder = CNNEncoder_(**kwargs)
        self.char_decoder = ResidualGenerator(**kwargs['char_kwargs'])
    def forward(self, x, characters=None, use_wd=True, add_end_mask=False, gen_chars=True):
        char_encoding = torch.cat([
            self.char_encoder(ch.transpose(1,2).squeeze(0).argmax(-1), use_wd=use_wd).mean(0,keepdim=True) # was .sum
            for ch in characters
        ])
        encoding = self.word_encoder(x, char_encoding, use_wd=use_wd, add_end_mask=add_end_mask)
        # producing words is used to regularize the model, 
        # but it is not necessarily useful for a lot of inference use cases
        if self.training and gen_chars:
            char_trees = [
                self.char_decoder(enc, size=l.shape[-1])
                for enc,l in zip(encoding.chunk(len(encoding)), characters)
            ]
            char_trees = [
                {
                    'tree': t,
                    'leaves': self.char_decoder.get_leaves(t),
                    'depths': self.char_decoder.get_leaves(t, attr='depth'),
                    'states': self.char_decoder.get_states(t)
                }
                for t in char_trees
            ]
        else:
            char_trees = None
        return encoding, char_trees
        
class CNNEncoder_(nn.Module):
    def __init__(self, *args, **kwargs):
        super(CNNEncoder_, self).__init__()
        i, h = kwargs['i'], kwargs['h']
        self.h = h
        self.wd = kwargs['wd']
        self.embed = wn(nn.Embedding(i, h))
        self.conv0 = wn(nn.Conv1d(h, h, 3, 1, 1))
        self.conv1 = wn(nn.Conv1d(h, h, 3, 1, 1))
        self.selu = nn.SELU()
        if kwargs.get('span_dropout') is not None:
            self.span_dropout = span_dropout

    def forward(self, x, characters=None, skip_embedding=False, use_wd=True, add_end_mask=False):
        if skip_embedding:
            e = x
        else:
            e = self.embed(x)
        if characters is not None:
            e = e + characters
        if self.training and self.wd and use_wd:
            if hasattr(self, 'span_dropout'):
                _e, (mask_ix, span_size) = self.span_dropout(e)
                e, masked = _e, e[mask_ix:mask_ix+span_size]
            else:
                e = word_dropout(e.unsqueeze(1), self.wd).squeeze(1)
                masked = None
        if add_end_mask:
            e = torch.cat([e, torch.zeros((2, self.h))])
        e = e.view(1, self.h, -1)
        c0 = self.conv0(self.selu(e))
        c1 = self.conv1(self.selu(c0))
        return c1.view(-1, self.h) + e.view(-1, self.h)


class Generator(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Generator, self).__init__()
        self.h = kwargs['h']
        self.i = kwargs['i']
        h = self.h
        i = self.i
        reduction = h // (lambda x: 1 if x is None else x)(kwargs.get('reduction'))
        if kwargs.get('char_level') is not None:
            self.V = wn(nn.Linear(h, kwargs['o']))
        self.reduction = reduction
        self.hidden = wn(nn.Linear(h, h))
        self.has_branches = wn(nn.Linear(reduction + h, 1))
        self.controller = nn.GRUCell(h, reduction)
        self.branch = wn(nn.Linear(h + reduction, h * 2))
        self.selu = nn.SELU()
        self._discretize = (lambda x: 0 if x is None else x)(kwargs.get('discretize'))

    def forward(self, x, size=None):
        if size is None:
            self.max_steps = get_n_branches(x)
        else:
            self.max_steps = get_n_branches(size)
        self.current_steps = 0
        return self.generate(x, is_root=True)

    def discretize(self, x):
        if self._discretize == 0:
            return x
        return torch.cat([
            x[:,:-self._discretize],
            x[:,-self._discretize:].softmax(-1),
        ], dim=-1)
        
    def generate(self, x, is_root=False, depth=0.):
        raise NotImplementedError
        
    def get_leaves(self, x, attr='terminal', cat=True, extra_attrs=[None], allow_none=False, partial_tree=False):
        def descend(x):
            if not partial_tree and x['left'] == {}:
                if allow_none:
                    leaves.append(x.get(attr))
                else:
                    leaves.append(x[attr])
                if type(extra_attrs) is list and extra_attrs[0] != None:
                    for a in extra_attrs:
                        leaves[-1] += ('_' + str(x[a]))
            elif partial_tree and x == {}:
                leaves.append(None)
            else:
                descend(x['left'])
                descend(x['right'])
        leaves = []
        descend(x)
        if cat:
            return torch.cat(leaves, dim=0)
        return leaves
    def attach_to_leaves(self, x, attachment, attachment_name='attachment', replace=False):
        attachment.reverse()
        def descend(x):
            if x['left'] == {} and attachment != []:
                if replace:
                    x.update(attachment.pop())
                else:
                    x[attachment_name] = attachment.pop()
            elif attachment == []:
                pass
            else:
                descend(x['left'])
                descend(x['right'])
        descend(x)
        return None
    def get_states(self, x, attr='state', leaves_only=False):
        def descend(x):
            if x['left'] == {}:
                leaves.append(x[attr])
            else:
                if not leaves_only:
                    leaves.append(x[attr])
                descend(x['left'])
                descend(x['right'])
        leaves = []
        descend(x)
        return torch.cat(leaves, dim=0)
    def get_subtrees(self, x, sent_len, attr='ix', cat=False, leaves_only=False):
        def descend(x):
            if x['left'] == {}:
                subtrees.append(x)
            else:
                if not leaves_only:
                    subtrees.append(x)
                descend(x['left'])
                descend(x['right'])
        subtrees = []    
        self.attach_to_leaves(x, list(range(sent_len)), 'ix')
        descend(x)
        return subtrees
    def compute_bottom_up_fx(self, tree, fx):
        def descend(x):
            if x['left'] == {}:
                return fx(torch.cat([x['h'], torch.zeros_like(x['h']), torch.zeros_like(x['h'])], dim=1))
            else:
                left = descend(x['left'])
                right = descend(x['right'])
                return fx(torch.cat([left, x['h'], right], dim=1))
        return descend(tree)
            
class ResidualGenerator(Generator):
    def __init__(self, *args, **kwargs):
        super(ResidualGenerator, self).__init__(**kwargs)
        self.h = kwargs['h']
        self.i = kwargs['i']
        h = self.h
        i = self.i
        reduction = h // (lambda x: 1 if x is None else x)(kwargs.get('reduction'))
        if kwargs.get('char_level') is not None:
            self.V = wn(nn.Linear(h, kwargs['o']))
        self.reduction = reduction
        self.controller = nn.GRUCell(h, reduction)
        self.has_branches = wn(nn.Linear(h + reduction, 1))
        self.branch = wn(nn.Linear(h + reduction, h * 2))
        self.selu = nn.SELU()
        self.hidden = wn(nn.Linear(h, h))
        self._discretize = (lambda x: 0 if x is None else x)(kwargs.get('discretize'))
    def generate(self, x, is_root=False, depth=0.):
        if is_root:
           x = self.discretize(self.selu(self.hidden(self.selu(x)))).mean(0, keepdim=True)
           h = torch.zeros((1, self.reduction))
        else:
           x, h = x
        h = self.controller(x, h)
        has_branches = self.has_branches(torch.cat([x, h], dim=-1))
        if self.training:
            has_branches = has_branches + torch.randn((1, 1))
        has_branches = has_branches.sigmoid()
        if has_branches < .5 or self.current_steps >= self.max_steps:
            return {
                'terminal': x if not hasattr(self, 'V') else self.V(x),
                'state': x,
                'h': h,
                'depth': depth + has_branches,
                'left': {},
                'right': {}
            }
        self.current_steps += 2
        left, right = self.selu(self.branch(torch.cat([x, h], dim=-1))).chunk(2, dim=-1)
        left = self.discretize(x - self.selu(self.hidden(left)))
        right = self.discretize(x - self.selu(self.hidden(right)))
        if random.choice([0, 1]):
            left_branch = self.generate([left, h], depth=depth + has_branches)
            right_branch = self.generate([right, h], depth=depth + has_branches)
            return {
                'state': x,
                'h': h,
                'left': left_branch,
                'right': right_branch
            }
        else:
            right_branch = self.generate([right, h], depth=depth + has_branches)
            left_branch = self.generate([left, h], depth=depth + has_branches)
            return {
                'state': x,
                'h': h,
                'left': left_branch,
                'right': right_branch
            }
            
class Copy(nn.Module):
    """ Allows model to learn to copy unknown words from input to output based on their index in the input """
    def __init__(self, *args, **kwargs):
        super(Copy, self).__init__()
        o, h = kwargs['i'], kwargs['h']
        limit, n_heads = kwargs['limit'], kwargs['n_copy_heads']
        self.h = h
        self.selu = nn.SELU()
        self.hidden = wn(nn.Linear(h, h))
        reduction = h // (lambda x: 1 if x is None else x)(kwargs.get('reduction'))        
        self.query0 = wn(nn.Conv1d(h + reduction, h, 3, 1, 1))
        self.query1 = wn(nn.Conv1d(h, h, 3, 1, 1))
        self.A = CustomMultiheadAttention(**{'h': h, 'n_heads':n_heads})
        self.V = wn(nn.Linear(h, o))
        self.C = wn(nn.Linear(h, limit))
        self._discretize = (lambda x: 0 if x is None else x)(kwargs.get('discretize'))
    def discretize(self, x):
        if self._discretize == 0:
            return x
        return torch.cat([
            x[:,:-self._discretize],
            x[:,-self._discretize:].softmax(-1),
        ], dim=-1)
    def forward(self, o, f):
        q = self.selu(self.query1(self.selu(self.query0(o.transpose(0,1).unsqueeze(0)))).transpose(1,2).transpose(0,1))
        kv = self.selu(f).unsqueeze(1) #self.discretize(self.selu(self.hidden(self.selu(f)))).unsqueeze(1)
        ao = self.selu(self.A(q, kv, sizes=[len(f)])[0].squeeze(1))
        out = torch.cat([self.V(ao), self.C(ao)], dim=-1)
        return out

class CustomMultiheadAttention(nn.Module):
    def __init__(self, *args, **kwargs):
        super(CustomMultiheadAttention, self).__init__()
        self.h, self.n_heads = kwargs['h'], kwargs['n_heads']
        self.head_dim = self.h // self.n_heads
        assert self.h == self.head_dim * self.n_heads
        self.out = wn(nn.Linear(self.h * 2, self.h))
    def forward(self, query, key, sizes=None, batch_first=False, return_scores_only=False, return_unnorm_scores=False, return_mix=False):
        if sizes is not None:
            sizes = torch.FloatTensor([
                [0 if i < s else 1 for i in range(key.shape[1 if batch_first else 0])]
                for s in sizes
            ])
        if not batch_first:
            query = query.transpose(0,1)
            key = key.transpose(0,1)
        b = query.shape[0]
        query = query.view(b, -1, self.n_heads, self.head_dim)
        key = key.view(b, -1, self.n_heads, self.head_dim)
        query = query.transpose(1,2)
        key = key.transpose(1,2)
        scores = torch.matmul(query, key.transpose(-2,-1)) / math.sqrt(self.head_dim)
        if sizes is not None:
            scores = scores.permute(1,2,0,3).masked_fill(sizes == 1, -1e9).permute(2,0,1,3)
        _scores = scores
        scores = scores.softmax(-1)
        mix = torch.matmul(scores, key).view(b, -1, self.h)
        #if return_mix:
        #    return mix.transpose(0,1), scores
        if not batch_first:
            mix = mix.transpose(0,1)
            query = query.transpose(0,1).contiguous().view(-1, b, self.h) 
        output = self.out(torch.cat([mix, query], dim=2))
        if return_unnorm_scores:
            return output, scores, _scores
        return output, scores

class Parser(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Parser, self).__init__()
        self.h, self.limit = kwargs['h'], kwargs['limit']
        self.selu = nn.SELU()
        self.encoder = CNNEncoder(**kwargs)
        self.decoder = ResidualGenerator(**kwargs)
        self.copy = Copy(**kwargs)
        #self.decoder.hidden.weight.data = self.copy.hidden.weight.data

        self.encoder.word_encoder.embed.weight.data = self.copy.V.weight.data
        self.encoder.char_encoder.embed.weight.data = self.encoder.char_decoder.V.weight.data
        self.encoder.word_encoder.conv0.weight.data = self.copy.query0.weight.data
        self.encoder.word_encoder.conv1.weight.data = self.copy.query1.weight.data
        
        #######
        #self.encoder.word_encoder.embed.bias.data = self.copy.V.bias.data
        #self.encoder.char_encoder.embed.bias.data = self.encoder.char_decoder.V.bias.data
        self.encoder.word_encoder.conv0.bias.data = self.copy.query0.bias.data
        self.encoder.word_encoder.conv1.bias.data = self.copy.query1.bias.data
        
        # tie word-level and char-level weights where possible
        self.decoder.hidden.weight.data = self.encoder.char_decoder.hidden.weight.data
        self.decoder.branch.weight.data = self.encoder.char_decoder.branch.weight.data
        self.decoder.has_branches.weight.data = self.encoder.char_decoder.has_branches.weight.data
        self.decoder.controller.weight_ih.data = self.encoder.char_decoder.controller.weight_ih.data
        self.decoder.controller.weight_hh.data = self.encoder.char_decoder.controller.weight_hh.data
        
        self.encoder.char_encoder.conv0.weight.data = self.encoder.word_encoder.conv0.weight.data
        self.encoder.char_encoder.conv1.weight.data = self.encoder.word_encoder.conv1.weight.data 
        
        #######
        self.decoder.hidden.bias.data = self.encoder.char_decoder.hidden.bias.data
        self.decoder.branch.bias.data = self.encoder.char_decoder.branch.bias.data
        self.decoder.has_branches.bias.data = self.encoder.char_decoder.has_branches.bias.data
        self.decoder.controller.bias_ih.data = self.encoder.char_decoder.controller.bias_ih.data
        self.decoder.controller.bias_hh.data = self.encoder.char_decoder.controller.bias_hh.data
        
        self.encoder.char_encoder.conv0.bias.data = self.encoder.word_encoder.conv0.bias.data
        self.encoder.char_encoder.conv1.bias.data = self.encoder.word_encoder.conv1.bias.data         
        
        self._discretize = (lambda x: 0 if x is None else x)(kwargs.get('discretize'))
        self.span_dropout = kwargs.get('span_dropout')
        self.out = wn(nn.Linear(self.h, 2))
        
    def discretize(self, x):
        if self._discretize == 0:
            return x
        return torch.cat([
            x[:,:-self._discretize],
            x[:,-self._discretize:].softmax(-1),
        ], dim=-1)

    def forward(self, x, characters=None, size=None, use_wd=True, skip_encoding=None, gen_chars=False):
        if skip_encoding is not None:
            encoding = x
        else:
            encoding, char_trees = self.encoder(x, characters, use_wd=use_wd, gen_chars=gen_chars)
        tree = self.decoder(encoding, size=size)
        hs = self.decoder.get_leaves(tree, attr='h')
        leaves = self.decoder.get_leaves(tree)
        if skip_encoding is not None:
            leaves_after_copy = self.copy(leaves, skip_encoding)
        else:
            leaves_after_copy = self.copy(torch.cat([leaves, hs], dim=-1), encoding)#
        depths = self.decoder.get_leaves(tree, attr='depth')
        states = self.decoder.get_states(tree)
        hs = self.decoder.get_states(tree, attr='h')
        return {
            'encoding': encoding,
            'tree': tree,
            'leaves': leaves,
            'leaves_after_copy': leaves_after_copy,
            'depths': depths,
            'states': states,
            'char_trees': char_trees,
            'hs': hs,
        }
