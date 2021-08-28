# -*- coding: utf-8 -*-
import os
import re
import _pickle as cPickle
from collections import Counter
import argparse
import multiprocessing
import math
import pdb
import random

import numpy as np
import pandas as pd
from nltk import word_tokenize

from preprocess import apply_parallel, get_tokens, get_group_df, get_vocab, filter_df

# +
np.random.seed(1234)
random.seed(1234)
UNK = '<unk>' # This has a vocab id, which is used to represent out-of-vocabulary words

category_dict = {'cloth': 'clothing', 'electronics': 'electronics', 'health_personal_care': 'health', 'home_kitchen': 'home',
                                   'reviews_clothing_shoes_and_jewelry': 'clothing', 'reviews_health_and_personal_care': 'health', 'reviews_electronics': 'electronics', 'reviews_home_and_kitchen': 'home'}

parser = argparse.ArgumentParser()

parser.add_argument('-seed', type=int, default=1234)
parser.add_argument('-n_processes', type=int, default=64) # for multiprocessing
parser.add_argument('-n_reviews', type=int, default=8)
parser.add_argument('-item_min_reviews', type=int, default=8)
parser.add_argument('-filter_doc_l', type=int, default=50)
parser.add_argument('-filter_sent_l', type=int, default=40)
parser.add_argument('-min_tf', type=int, default=16)

parser.add_argument('-data', type=str, default='amazon')
parser.add_argument('-n_per_item', type=int, default=2)
parser.add_argument('-dir_train', type=str, default='data/amazon/train')
parser.add_argument('-path_dev', type=str, default='data/amazon/dev.csv')
parser.add_argument('-path_test', type=str, default='data/amazon/test.csv')
    
parser.add_argument('-name_output', type=str, default='data_df.pkl')
parser.add_argument('-name_vocab', type=str, default='vocab.pkl')
config = parser.parse_args()

config.path_output = os.path.join('data', config.data, config.name_output)
config.path_vocab = os.path.join('data', config.data, config.name_vocab)


# +
def get_amazon_ref_df(path_ref):
    ref_raw_df = pd.read_csv(path_ref, sep='\t')
    ref_list = []
    for _, row in ref_raw_df.iterrows():
        business_id = row['prod_id']
        category = category_dict[row['cat']]
        text_list = [row['rev%i'%(i_rev+1)] for i_rev in range(8)]
        tokens = [sent_tokens for text in text_list for sent_tokens in get_tokens(text)]
        text = ' </DOC> '.join(text_list)
        stars = [0 for _ in range(8)]
        doc_l = len(tokens)
        sent_l = [len(line) for line in tokens]
        max_sent_l = max(sent_l)

        for i_summ in range(3):
            summary = row['summ%i'%(i_summ+1)]
            summary_tokens = get_tokens(summary)
            summary_doc_l = len(summary_tokens)
            summary_max_sent_l = max([len(line) for line in tokens])

            ref_list.append({'business_id': business_id, 'category': category, 'text': text, 'tokens': tokens, 'summary': summary, 'stars': stars, \
                                         'doc_l': doc_l, 'max_sent_l':max_sent_l, 'sent_l':sent_l,
                                         'summary_tokens': summary_tokens, 'summary_doc_l': summary_doc_l, 'summary_max_sent_l': summary_max_sent_l})

    ref_df = pd.DataFrame(ref_list)
    return ref_df

def get_amazon_raw_df(data_paths):
    data_raw_dfs = []
    for data_path in data_paths:
        item_raw_df = pd.read_csv(data_path, '\t')
        item_raw_df['tokens'] = item_raw_df['review_text'].apply(get_tokens)
        item_raw_df = item_raw_df[item_raw_df['tokens'].apply(lambda tokens: len(tokens) > 2)]
        if 'None' in list(item_raw_df.rating.values):
            item_raw_df = item_raw_df[item_raw_df['rating'].apply(lambda rating: rating != 'None')]
            item_raw_df['rating'] = item_raw_df['rating'].apply(lambda r: float(r))
        if len(item_raw_df) == 0: continue
        data_raw_df = item_raw_df.groupby('group_id').agg({
            'category': lambda category_list: category_dict[list(category_list)[0]],
            'tokens': lambda tokens_list: list(tokens_list),
            'rating': lambda stars_list: list(stars_list)
        })
        data_raw_dfs.append(data_raw_df)
    data_raw_df = pd.concat(data_raw_dfs)
    return data_raw_df

def get_amazon_df(config):
    dev_df = get_amazon_ref_df(config.path_dev)
    test_df = get_amazon_ref_df(config.path_test)

    train_paths = [os.path.join(config.dir_train, data_name) for data_name in os.listdir(config.dir_train)]
    train_raw_df = apply_parallel(train_paths, n_processes=32, map_func=get_amazon_raw_df).reset_index().rename(columns={'group_id': 'business_id', 'rating': 'stars'})

    train_df = get_group_df(train_raw_df, n_reviews=config.n_reviews, filter_sent_l=config.filter_sent_l, filter_doc_l=config.filter_doc_l, \
                            item_min_reviews=config.item_min_reviews, n_per_item=config.n_per_item)

    return train_df, dev_df, test_df

def apply_token_idxs(tokens_series):
    def get_token_idxs(tokens):
        return [[word_to_idx[token] if token in word_to_idx else word_to_idx[UNK] for token in sent] for sent in tokens]
    return tokens_series.apply(get_token_idxs)


# -

if __name__ == '__main__':
    if config.data == 'yelp':
        train_df, dev_df, test_df = get_yelp_df(config)
    elif config.data == 'amazon':
        train_df, dev_df, test_df = get_amazon_df(config)
    
    word_to_idx = cPickle.load(open(config.path_vocab, 'rb'))
#     word_to_idx = get_vocab(train_df, config)
#     cPickle.dump(word_to_idx, open(config.path_vocab, 'wb'))
    
    train_df['token_idxs'] = apply_parallel(train_df['tokens'], n_processes=config.n_processes, map_func=apply_token_idxs)
    dev_df['token_idxs'] = apply_parallel(dev_df['tokens'], n_processes=config.n_processes, map_func=apply_token_idxs)
    test_df['token_idxs'] = apply_token_idxs(test_df['tokens'])
    
    train_df = filter_df(train_df)
    dev_df = filter_df(dev_df, summary=True)
    test_df = filter_df(test_df, summary=True)
    
    print('saving preprocessed dataframe into %s ...'%config.path_output)
    cPickle.dump((train_df, dev_df, test_df), open(config.path_output, 'wb'))
