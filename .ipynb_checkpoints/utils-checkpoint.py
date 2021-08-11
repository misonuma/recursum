# -*- coding: utf-8 -*-
# +
import re
import random
import multiprocessing

import numpy as np
import pandas as pd
from nltk import word_tokenize


# -


def apply_parallel(datas, num_split, map_func):
    p = multiprocessing.Pool(processes=num_split)
    data_split = np.array_split(datas, num_split)
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


def filter_words(word_tf_dict, min_tf=None, stop_words=[]):
    filtered_word_tf_dict = dict([(word, tf) for word, tf in word_tf_dict if word not in stop_words])
    if min_tf: filtered_word_tf_dict = {word: tf for word, tf in filtered_word_tf_dict.items() if tf >= min_tf}
    filtered_words = [word for word, _ in filtered_word_tf_dict.items()]        
    return filtered_words


def apply_token_idxs(tokens_series):
    def get_token_idxs(tokens):
        return [[word_to_idx[token] if token in word_to_idx else word_to_idx[UNK] for token in sent] for sent in tokens]
    return tokens_series.apply(get_token_idxs)
