import streamlit as st
import pandas as pd
import numpy as np
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import plotly.graph_objects as go
from igraph import Graph, EdgeSeq
import pickle
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

def size_fxn(context, x):
    return len(x)

def target_fxn(context, x):
    return torch.LongTensor([len(context['vocab']) - (x + 1)])

def preprocess_sentiment(x):
    return [x]

SAVE_DIR = '06_12_01'

with open(f'./saved_models/{SAVE_DIR}/model_configs.pkl','rb') as f:
    model_configs = pickle.load(f)
    model_config = model_configs['model']
    classifier_config = model_configs['classifier']
    copy_config = model_configs['copy']

ds = LanguageDataset({ 'state_dict': torch.load(f'./saved_models/{SAVE_DIR}/data.pt') })
E = Encoder(**model_config)
G = Generator(**model_config)
C = Copy(**copy_config)
CL = Classifier(**classifier_config)

E.load_state_dict(torch.load(f'./saved_models/{SAVE_DIR}/models/E.pt'))
G.load_state_dict(torch.load(f'./saved_models/{SAVE_DIR}/models/G.pt'))
C.load_state_dict(torch.load(f'./saved_models/{SAVE_DIR}/models/C.pt'))
CL.load_state_dict(torch.load(f'./saved_models/{SAVE_DIR}/models/CL.pt'))

E.eval(); G.eval(); C.eval(); CL.eval()

txt = st.text_input('Enter a sentence...',"@usairways my dog is very happy! thanks guys!")

words_data = ds.preprocess_new_observations('text', pd.Series([txt]))
sentiment_data = ds.preprocess_new_observations('airline_sentiment', pd.Series(['positive']))

features, _ = E(batch_data([words_data['vectors'][0]], 0, C.limit))
tree = G(G.act(features.sum(0)), sizes=[C.limit], return_trees=True)[0]
leaves = batch_data([G.get_leaves(tree)], 0, C.limit)
leaves = C(leaves, G.act(features), sizes=[len(words_data['vectors'][0])])
tree = attach_to_leaves(tree, leaves, ds.vars['text'], model_config['io'], G, words_data['text'][0])
print_tree(tree, lambda x: x, 'attachment')
edges = G.add_vertices(tree)

_leaves = G.get_leaves(tree, attr='attachment', cat=False, extra_attrs=['vertex'])
_leaves = {x.split('_')[1]:x.split('_')[0] for x in _leaves}
#_leaves
#edges = [[0, 1], [1, 2], [1, 3], [0, 4]]
nr_vertices = max(list(chain(*edges))) + 1
v_label = list(map(str, range(nr_vertices)))
_G = Graph(edges) # 2 stands for children number
lay = _G.layout('rt', root=(0,0))

scale = 1
position = {k: list(map(lambda x: x * scale, lay[k])) for k in range(nr_vertices)}
Y = [list(map(lambda x: x * scale, lay[k]))[1] for k in range(nr_vertices)]
M = max(Y)

es = EdgeSeq(_G) # sequence of edges
_E = [e.tuple for e in _G.es] # list of edges

L = len(position)
Xn = [position[k][0] for k in range(L)]
Yn = [2*M-position[k][1] for k in range(L)]
Xe = []
Ye = []
for edge in _E:
    Xe+=[position[edge[0]][0],position[edge[1]][0], None]
    Ye+=[2*M-position[edge[0]][1],2*M-position[edge[1]][1], None]

labels = v_label
#labels
#_leaves
labels = ['' if l not in _leaves else _leaves[str(l)] for l in labels]
#labels

layout = go.Layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    showlegend=False,
    xaxis={'visible': False},
    yaxis={'visible': False},
    margin=dict(l=.1, r=.1, t=.1, b=.1),
    width=2000,
    height=700,
)

fig = go.Figure(layout=layout)
fig.add_trace(go.Scatter(x=Xe,
                   y=Ye,
                   mode='lines',
                   line=dict(color='rgb(210,210,210)', width=1),
                   hoverinfo='none'
                   ))
fig.add_trace(go.Scatter(x=Xn,
                  y=Yn,
                  mode='text',
                  name='bla',
                  marker=dict(symbol='circle-dot',
                                size=5,
                                color='#6175c1',    #'#DB4551',
                                line=dict(color='rgb(50,50,50)', width=1)
                                ),
                  text=labels,
                  hoverinfo='none',
                  opacity=.9,
                  textposition="bottom center",
                  ))

st.plotly_chart(fig, use_container_width=True)
