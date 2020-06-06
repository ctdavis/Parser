import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchcontrib.nn as tcnn
import re
import random
import math

from utils import *

class BatchGenerator(nn.Module):
    def __init__(self, *args, **kwargs):
        super(BatchGenerator, self).__init__()
        self.pad = kwargs['pad']
    def forward(self, x, sizes, return_trees=False, context=None):
        trees = []
        if context is None:
            for size, enc in zip(sizes, x.chunk(x.shape[0])):
                self.source_size = size
                self.max_steps = get_n_branches(size)
                self.current_steps = 0
                trees += [self.generate(enc, is_root=True)]
        else:
            for size, enc, con in zip(sizes, x.chunk(x.shape[0]), context.chunk(context.shape[1], dim=1)):
                self.source_size = size
                self.max_steps = get_n_branches(size)
                self.current_steps = 0
                trees += [self.generate(enc, con, is_root=True)]
        if return_trees:
            return trees
        else:
            leaves = nn.utils.rnn.pad_sequence(
                [self.get_leaves(t) for t in trees],
                padding_value=0
            )
            leaves = define_padded_vectors(leaves, self.pad)      
            return leaves
    def generate(self, x, is_root=True, *args, **kwargs):
        raise NotImplementedError
    def get_leaves(self, x, attr='terminal', cat=True):
        def descend(x):
            if x['left'] == {}:
                leaves.append(x[attr])
            else:
                descend(x['left'])
                descend(x['right'])
        leaves = []
        descend(x)
        if cat:
            return torch.cat(leaves, dim=0)
        return leaves
    def attach_to_leaves(self, x, attachment):
        attachment.reverse()
        def descend(x):
            if x['left'] == {} and attachment != []:
                x['attachment'] = attachment.pop()
            elif x['left'] == {} and attachment == []:
                pass
            else:
                descend(x['left'])
                descend(x['right'])
        descend(x)
        return None

    def get_states(self, x, leaves_only=False):
        def descend(x):
            if x['left'] == {}:
                leaves.append(x['state'])
            else:
                if not leaves_only:
                    leaves.append(x['state'])
                descend(x['left'])
                descend(x['right'])
        leaves = []
        descend(x)
        return torch.cat(leaves, dim=0)
    def get_leaves_from_subtrees(self, x, attr='terminal', cat=True):
        def descend(x):
            if x['left'] == {}:
                leaves.append(x[attr])
            else:
                leaves.append(self.get_leaves(x, attr, cat))
                descend(x['left'])
                descend(x['right'])
        leaves = []
        descend(x)
        return leaves

class CharEncoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super(CharEncoder, self).__init__()
        i, h = kwargs['i'], kwargs['h']
        self.act = kwargs['act']
        self.conv0 = nn.Conv1d(i, h, 3, 1, 1)
        self.conv1 = nn.Conv1d(h, h, 3, 1, 1)
    def forward(self, x):
        c0 = self.act(self.conv0(x))
        c1 = self.act(self.conv1(c0))
        c1 = torch.cat([c0, c1], dim=1)
        return c1.mean(-1)

class Attention(nn.Module):
    def __init__(self, dim, sm_over_context=True, double_linear_out=False):
        super(Attention, self).__init__()
        if double_linear_out is True:
            self.double_linear_out = True
            self.linear_out = nn.Linear(dim*2, dim*2)
        else:
            self.double_linear_out = False
            self.linear_out = nn.Linear(dim*2, dim)
        self.sm_over_context = sm_over_context
    def forward(self, output, context, sizes=None):
        if sizes is not None:
            sizes = torch.FloatTensor([
                [0 if i < s else 1 for i in range(context.shape[0])]
                for s in sizes
            ])
        n_dims = output.dim()
        if n_dims == 3:
            output = output.transpose(0,1) # L x B x D -> B x L x D
            context = context.transpose(0,1)
        else:
            output = output.unsqueeze(0) # L x D -> B x L x D 
            context = context.unsqueeze(0) # L x D -> B x L x D
        batch_size = output.size(0)
        hidden_size = output.size(2)
        input_size = context.size(1)
        out_size = output.size(1)
        # (batch, out_len, dim) * (batch, in_len, dim) -> (batch, out_len, in_len)
        attn = torch.bmm(output, context.transpose(1, 2))
        if sizes is not None:
            attn = attn.transpose(0,1).masked_fill(sizes.bool(), False).transpose(0,1)
            mask = ((attn != 0.).float() - 1) * 9999
            attn = (attn + mask).contiguous()
            
        attn = F.softmax(
            attn.view(-1, input_size if self.sm_over_context else out_size),
            dim=1
        ).view(batch_size, -1, input_size)
     
        # (batch, out_len, in_len) * (batch, in_len, dim) -> (batch, out_len, dim)
        mix = torch.bmm(attn, context)

        # concat -> (batch, out_len, 2*dim)
        combined = torch.cat((mix, output), dim=2)

        # output -> (batch, out_len, dim)
        output = self.linear_out(combined.view(-1, 2 * hidden_size))\
            .view(batch_size, -1, hidden_size * 2 if self.double_linear_out else hidden_size)
        if n_dims == 3:
            output = output.transpose(0,1)
        else:
            output = output.squeeze(0)
        return output, attn

class MultiheadAttention(nn.Module):
    def __init__(self, *args, **kwargs):
        super(MultiheadAttention, self).__init__()
        self.h, self.n_heads = kwargs['h'], kwargs['n_heads']
        #self.act = kwargs['act']
        self.head_dim = self.h // self.n_heads
        assert self.h == self.head_dim * self.n_heads
        self.out = nn.Linear(self.h * 2, self.h)
    def forward(self, query, key, sizes=None, batch_first=False):
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
        #print(scores.shape, sizes.shape)
        if sizes is not None:
            scores = scores.permute(1,2,0,3).masked_fill(sizes == 1, -1e9).permute(2,0,1,3)
        #print(scores)
        scores = scores.softmax(-1)
        mix = torch.matmul(scores, key).view(b, -1, self.h)
        if not batch_first:
            mix = mix.transpose(0,1)
            query = query.transpose(0,1).contiguous().view(-1, b, self.h) 
        output = self.out(torch.cat([mix, query], dim=2))
        return output, scores

class Generator(BatchGenerator):
    def __init__(self, *args, **kwargs):
        super(Generator, self).__init__(*args, **kwargs)
        o, h = kwargs['io'], kwargs['h']
        self.act = kwargs['act']
        self.o = o
        self.h = h
        self.has_branches = nn.Linear(h, 1)
        self.gamma_branch = nn.Linear(h, h * 2)
        self.beta_branch = nn.Linear(h, h * 2)
        self.film = tcnn.FiLM()
        if kwargs.get('dr') is not None:
            self.dropout = nn.AlphaDropout(kwargs['dr'])

    def generate(self, x, is_root=False, depth_cost=torch.FloatTensor([0.]), unk=1, *args, **kwargs):
        if is_root:
            self.global_state = x
            x = torch.zeros_like(self.global_state)
            #gamma, beta = self.condition(self.act(x)).chunk(2, dim=-1)#self.gamma(self.act(x)), self.beta(self.act(x))
            _, gamma = self.act(self.gamma_branch(x)).chunk(2, dim=1)
            _, beta = self.act(self.beta_branch(x)).chunk(2, dim=1)
            x = self.act(self.film(self.global_state, gamma, beta))
        if hasattr(self, 'dropout') and self.training:
            x = self.dropout(x)
        has_branches = self.has_branches(x).sigmoid()
        if (has_branches.item() < .5) or (self.current_steps >= self.max_steps):
            return {
                'terminal': x,
                'left': {},
                'right': {},
                'depth': depth_cost + has_branches,
                'state': x,
            }
        else:
            self.current_steps += 2
            left_gamma, right_gamma = self.act(self.gamma_branch(x)).chunk(2, dim=1)
            left_beta, right_beta = self.act(self.beta_branch(x)).chunk(2, dim=1)
            left = self.act(self.film(self.global_state, left_gamma, left_beta))
            right = self.act(self.film(self.global_state, right_gamma, right_beta))
            if random.choice([True, False]):
                left_branch = self.generate(left, depth_cost=depth_cost+has_branches, *args, **kwargs)
                right_branch = self.generate(right, depth_cost=depth_cost+has_branches, *args, **kwargs)
            else:
                right_branch = self.generate(right, depth_cost=depth_cost+has_branches, *args, **kwargs)
                left_branch = self.generate(left, depth_cost=depth_cost+has_branches, *args, **kwargs)
            return {
                'state': x,
                'left': left_branch,
                'right': right_branch,
            }

class ResidualGenerator(BatchGenerator):
    def __init__(self, *args, **kwargs):
        super(ResidualGenerator, self).__init__(*args, **kwargs)
        o, h = kwargs['io'], kwargs['h']
        self.act = kwargs['act']
        self.o = o
        self.h = h
        self.has_branches = nn.Linear(h, 1)
        self.hidden = nn.Linear(h, h)
        #self.gamma_branch = nn.Linear(h, h * 2)
        #self.beta_branch = nn.Linear(h, h * 2)

        #self.gamma = nn.Linear(h, h)
        #self.beta = nn.Linear(h, h)
        #self.condition = nn.Linear(h, h * 2)
        self.branch = nn.Linear(h, h * 2)

        #self.film = tcnn.FiLM()
        if kwargs.get('dr') is not None:
            self.dropout = nn.AlphaDropout(kwargs['dr'])

    def generate(self, x, is_root=False, depth_cost=torch.FloatTensor([0.]), unk=1, *args, **kwargs):
        #if is_root:
        #    self.global_state = x
        #    x = torch.zeros_like(self.global_state)
        #    #gamma, beta = self.condition(self.act(x)).chunk(2, dim=-1)#self.gamma(self.act(x)), self.beta(self.act(x))
        #    _, gamma = self.act(self.gamma_branch(x)).chunk(2, dim=1)
        #    _, beta = self.act(self.beta_branch(x)).chunk(2, dim=1)
        #    x = self.act(self.film(self.global_state, gamma, beta))
        if hasattr(self, 'dropout') and self.training:
            x = self.dropout(x)
        has_branches = self.has_branches(x).sigmoid()
        if (has_branches.item() < .5) or (self.current_steps >= self.max_steps):
            return {
                'terminal': x,
                'left': {},
                'right': {},
                'depth': depth_cost + has_branches,
                'state': x,
            }
        else:
            self.current_steps += 2
            left, right = self.branch(x).chunk(2, dim=1)
            #left_gamma, right_gamma = self.act(self.gamma_branch(x)).chunk(2, dim=1)
            #left_beta, right_beta = self.act(self.beta_branch(x)).chunk(2, dim=1)
            left, right = x - self.act(self.hidden(self.act(left))), x - self.act(self.hidden(self.act(right)))
            #left_gamma, left_beta = self.condition(self.act(left)).chunk(2, dim=-1) #self.gamma(self.act(left)), self.beta(self.act(left))
            #right_gamma, right_beta = self.condition(self.act(right)).chunk(2, dim=-1) #self.gamma(self.act(right)), self.beta(self.act(right))
            
            #left = self.act(self.film(self.global_state, left_gamma, left_beta))
            #right = self.act(self.film(self.global_state, right_gamma, right_beta))
            if random.choice([True, False]):
                left_branch = self.generate(left, depth_cost=depth_cost+has_branches, *args, **kwargs)
                right_branch = self.generate(right, depth_cost=depth_cost+has_branches, *args, **kwargs)
            else:
                right_branch = self.generate(right, depth_cost=depth_cost+has_branches, *args, **kwargs)
                left_branch = self.generate(left, depth_cost=depth_cost+has_branches, *args, **kwargs)
            return {
                'state': x,
                'left': left_branch,
                'right': right_branch,
            }

class Encoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Encoder, self).__init__()
        io, h = kwargs['io'], kwargs['h']
        self.act = kwargs['act']
        self.batch = kwargs['batch']
        self.wd = kwargs['wd']
        self.h = h
        self.embed = nn.Embedding(io, h)
        self.conv1 = nn.Conv1d(h, h, 3, 1, 1)
        self.conv2 = nn.Conv1d(h, h, 3, 1, 1)
        self.conv3 = nn.Conv1d(h, h, 3, padding=(2*3 - 3 - (3-1)*(2-1)) + 1, dilation=2) 
        self.conv4 = nn.Conv1d(h, h,  3, padding=(2*3 - 3 - (3-1)*(2-1)) + 1, dilation=2) 
        if kwargs.get('adaptor'):
            self.adaptor = nn.Linear(h, kwargs['adaptor'])  
    def forward(self, x, aux=None):
        if self.training and self.wd is not None:
            e = self.embed(x)
            if aux is not None:
                e = e + aux
            e, rw = word_dropout(e, dropout=self.wd)
        else:
            e = self.embed(x)
            if aux is not None:
                e = (e + aux)
            rw = None
        e = e.transpose(0,1).transpose(1,2)
        c1 = self.conv1(self.act(e))
        c2 = self.conv2(self.act(c1)) + e
        c3 = self.conv3(self.act(c2)) + c1
        c4 = self.conv4(self.act(c3)) + c2
        if aux is None:
            features = (c4).transpose(1,2).transpose(0,1)
            if hasattr(self, 'adaptor'):
                features = self.adaptor(self.act(features.mean(0)))
            else:
                features = features#.sum(0)
        else:
            features = (c4).transpose(1,2).transpose(0,1)#.sum(0)
        return features, rw

class Classifier(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Classifier, self).__init__()
        self.io, self.h = kwargs['io'], kwargs['h']
        self.act = kwargs['act']
        if kwargs.get('n_heads') is not None:
            self.A = MultiheadAttention(**{'h': self.h, 'n_heads': kwargs['n_heads'] })
        else:
            self.A = Attention(self.h)
        self.C = nn.Linear(self.io * self.h, self.io)
    def forward(self, states, output_set):
        output, attn = zip(*[
            self.A(
                output_set,
                state,
            )
            for state in states
        ])
        return torch.cat([self.C(self.act(o).view(1, -1)) for o in output]), attn

class Copy(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Copy, self).__init__()
        self.io, self.h = kwargs['io'], kwargs['h']
        self.act = kwargs['act']
        self.limit = kwargs['limit']
        if kwargs.get('n_heads') is not None:
            self.A = MultiheadAttention(**{'h': self.h, 'n_heads': kwargs['n_heads'] })
        else:
            self.A = Attention(self.h)
        self.V = nn.Linear(self.h, self.io)
        self.C = nn.Linear(self.h, self.limit)
    def forward(self, output, features, sizes):
        attn_output = self.act(self.A(output, features, sizes)[0])
        output = torch.cat([self.V(attn_output), self.C(attn_output)], dim=-1)
        return output
