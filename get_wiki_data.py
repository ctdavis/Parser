from lxml import etree
import requests
import pandas as pd
import time
import pickle
from datetime import datetime
import re

from utils import *

def get_data():
    data = requests.get('https://en.wikipedia.org/wiki/Special:Random')
    data = etree.fromstring(data.text, etree.HTMLParser())
    return ''.join(data.xpath('//p//text()')).replace('\n',' ')

def get_vocab(preprocessor, output_file=None, n_samples=100, sleep=0.01, limit=30):
    data = []
    for _ in range(n_samples):
        data += [get_data()]
        time.sleep(sleep)
        print(data)
    data = pd.DataFrame({ 'text': data })
    config = {
        'df': data,
        'limit': limit,
        'vrs': {
            'text': {
                'reference': 'text',
                'vocab_min_freq': 30,
                'char_min_freq': 15,
                'rank': .05,
                'preprocessor': preprocessor,
                'base_vocab': { '<pad>': 0, '<unk>': 1 },
            }
        }
    }
    ds = Dataset(config)
    if output_file:
        with open(output_file, 'wb') as f:
            pickle.dump(ds.vrs['text']['V'], f)
        with open('chars_' + output_file, 'wb') as f:
            pickle.dump(ds.vrs['text']['cV'], f)
    return ds, data

#if __name__ == "__main__":
#    get_vocab(preprocessor, 'random_wiki_' + datetime.today().strftime('%m/%d/%Y') + '.pkl')


