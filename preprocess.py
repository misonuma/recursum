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

np.random.seed(1234)
random.seed(1234)

# special tokens
PAD = '<pad>' # This has a vocab id, which is used to pad the encoder input, decoder input and target sequence
UNK = '<unk>' # This has a vocab id, which is used to represent out-of-vocabulary words
BOS = '<p>' # This has a vocab id, which is used at the beginning of every decoder input sequence
EOS = '</p>' # This has a vocab id, which is used at the end of untruncated target sequences


def apply_parallel(datas, n_processes, map_func):
    p = multiprocessing.Pool(processes=n_processes)
    data_split = np.array_split(datas, n_processes)
    output_dfs = p.map(map_func, data_split)
    p.close()
    output_df = pd.concat(output_dfs)
    return output_df


def get_tokens(text):
    def preprocess(text):
        text = text.lower()
        text = re.sub(r'[0-9]+.[0-9]+|[0-9]+,[0-9]+|[0-9]+', '#', text)
        text = re.sub('-', ' ', text)
        code_regex = re.compile('["\%&\\\\\()*+/:;<=>?@[\\]^_`{|}~「」〔〕“”〈〉『』【】＆＊・（）＄＠。、？！｀＋￥％]|-lrb-(.*?)-rrb-|-lsb-(.*?)-rsb-')
        text = code_regex.sub('', text)
        return text

    tokenize = lambda lines: [word_tokenize(line) for line in lines]
    filtering = lambda tokens: [line_tokens for line_tokens in tokens if len(line_tokens) > 2]

    preprocessed_text = preprocess(text)
    lines = [line.strip() for line in re.split('[.!]', preprocessed_text) if not line == '']
    tokens = tokenize(lines)
    tokens = filtering(tokens)
    return tokens


def get_group_df(group_tmp_df, n_reviews, filter_sent_l, filter_doc_l, item_min_reviews, n_per_item):
    print('Each item will appear {} times'.format(n_per_item))

    batch_list = []
    for _, row in group_tmp_df.iterrows():
        tokens_stars_list = [(sents, stars) for sents, stars in zip(row.tokens, row.stars) if max([len(sent) for sent in sents]) <= filter_sent_l] # filter
        if len(tokens_stars_list) < item_min_reviews: continue
        i_per_item = 0
        while i_per_item < n_per_item:
            batch_tokens_stars = random.sample(tokens_stars_list, n_reviews)
            tokens = [sent for sents, _ in batch_tokens_stars for sent in sents]
            doc_l = len(tokens)
            sent_l = [len(sent) for sent in tokens]
            max_sent_l = max(sent_l)
            stars = [star for _, star in batch_tokens_stars]
            if doc_l <= filter_doc_l:
                batch_list.append({'business_id': row.business_id, 'tokens': tokens, 'doc_l': doc_l, 'sent_l': sent_l, 'max_sent_l': max_sent_l, 'stars': stars})
                i_per_item += 1
    
    group_df = pd.DataFrame(batch_list)
    return group_df


def get_vocab(train_df, config):
    def filter_words(word_tf_dict, min_tf=None, stop_words=[]):
        filtered_word_tf_dict = dict([(word, tf) for word, tf in word_tf_dict if word not in stop_words])
        if min_tf: filtered_word_tf_dict = {word: tf for word, tf in filtered_word_tf_dict.items() if tf >= min_tf}
        filtered_words = [word for word, _ in filtered_word_tf_dict.items()]        
        return filtered_words
    
    words_list = train_df['tokens'].apply(lambda tokens: [token for line in tokens for token in line])
    word_tf_dict = sorted(Counter([word for words in words_list for word in words]).items(), key=lambda x: x[1])
    
    special_words = [PAD, UNK, BOS, EOS]
    lm_words = special_words + filter_words(word_tf_dict, min_tf=config.min_tf)
    word_to_idx = {word: idx for idx, word in enumerate(lm_words)}
    
    return word_to_idx


def filter_df(data_df, summary=False):
    save_df = data_df[['business_id', 'doc_l', 'sent_l', 'max_sent_l', 'token_idxs', 'tokens']]
    if summary: save_df = data_df[['business_id', 'doc_l', 'sent_l', 'max_sent_l', 'tokens', 'token_idxs', \
                                                               'text', 'summary', 'summary_tokens', 'summary_doc_l', 'summary_max_sent_l']]
    return save_df
