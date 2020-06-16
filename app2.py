import streamlit as st
import torch
import torch.nn.functional as F
import pandas as pd

def swish_fx(x):
    return x * torch.sigmoid(x)

def mish_fx(x):
    return x * torch.tanh(F.softplus(x))

def tsish_fx(x):
    return x * torch.sigmoid(F.selu(x))

def create_df(x, act):
    df = pd.DataFrame(torch.cat([x.unsqueeze(1), act(x).unsqueeze(1)], dim=1).tolist(), columns=['x','f(x)'])
    df.set_index('x', inplace=True)
    return df

x = torch.FloatTensor(list(range(-2, 2)))

selu = create_df(x, F.selu)
swish = create_df(x, swish_fx)
mish = create_df(x, mish_fx)
tsish = create_df(x, tsish_fx)
selu_swish = create_df(x, lambda x: swish_fx(F.selu(x)))
selu_mish = create_df(x, lambda x: mish_fx(F.selu(x)))

'selu'
st.line_chart(selu)
'swish'
st.line_chart(swish)
'mish'
st.line_chart(mish)
'selu_swish'
st.line_chart(selu_swish)
'selu_mish'
st.line_chart(selu_mish)
'tsish'
st.line_chart(tsish)
