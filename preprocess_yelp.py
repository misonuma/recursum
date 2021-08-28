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

parser = argparse.ArgumentParser()

parser.add_argument('-seed', type=int, default=1234)
parser.add_argument('-n_processes', type=int, default=64) # for multiprocessing
parser.add_argument('-n_reviews', type=int, default=8)
parser.add_argument('-item_min_reviews', type=int, default=8)
parser.add_argument('-filter_doc_l', type=int, default=50)
parser.add_argument('-filter_sent_l', type=int, default=40)
parser.add_argument('-min_tf', type=int, default=16)

parser.add_argument('-data', type=str, default='yelp')
parser.add_argument('-n_per_item', type=int, default=12)
parser.add_argument('-dir_train', type=str, default='data/yelp/train')
parser.add_argument('-dir_val', type=str, default='data/yelp/val')
parser.add_argument('-dir_test', type=str, default='data/yelp/test')
parser.add_argument('-path_ref', type=str, default='data/yelp/references.csv')

parser.add_argument('-name_output', type=str, default='data_df.pkl')
parser.add_argument('-name_vocab', type=str, default='vocab.pkl')
config = parser.parse_args()

config.path_output = os.path.join('data', config.data, config.name_output)
config.path_vocab = os.path.join('data', config.data, config.name_vocab)


# +
def get_yelp_tmp_df(data_paths):
    data_raw_dfs = []
    for data_path in data_paths:
        f = open(data_path, 'r')
        item_raw_df = pd.read_json(f)
        f.close()
        data_raw_dfs.append(item_raw_df)
    data_raw_df = pd.concat(data_raw_dfs)
    data_raw_df['tokens'] = data_raw_df['text'].apply(get_tokens)
    data_raw_df = data_raw_df[data_raw_df['tokens'].apply(lambda tokens: len(tokens) > 2)]
    return data_raw_df

def get_yelp_ref_df(path_ref, train_tmp_df, dev_tmp_df, test_tmp_df):
    def get_review_business_id_dict(ref_tmp_df, train_tmp_df, dev_tmp_df, test_tmp_df):
        ref_review_ids = []
        for _, row in ref_tmp_df.iterrows():
            ref_review_ids += [row['Input.original_review_%i_id' % i] for i in range(config.n_reviews)]
        ref_review_id_df = pd.DataFrame(ref_review_ids, columns=['review_id']) # only review_id in reference.csv
        concat_raw_df = pd.concat([train_tmp_df, dev_tmp_df, test_tmp_df])[['review_id', 'business_id', 'stars']] # filter review_id and business_id pair
        review_business_id_dict = {row.review_id: row.business_id for _, row in pd.merge(ref_review_id_df, concat_raw_df).iterrows()}
        review_stars_dict = {row.review_id: row.stars for _, row in pd.merge(ref_review_id_df, concat_raw_df).iterrows()}
        return review_business_id_dict, review_stars_dict

    ref_tmp_df = pd.read_csv(path_ref)
    ref_tmp_df['business_id_csv'] = ref_tmp_df.apply(lambda row: row['Input.business_id'] if row['Input.business_id'] != '#NAME?' else 'null_%i' % row.name, axis=1)
    review_business_id_dict, review_stars_dict = get_review_business_id_dict(ref_tmp_df, train_tmp_df, dev_tmp_df, test_tmp_df)

    ref_raw_dfs = []
    for index, row in ref_tmp_df.iterrows():
        business_id = None 
        texts, review_ids, stars = [], [], []
        summary = row['Answer.summary']
        for i in range(config.n_reviews):
            text = row['Input.original_review_%i' % i]
            review_id = row['Input.original_review_%i_id' % i]
            star = review_stars_dict[review_id] if review_id in review_stars_dict else 3

            texts.append(text)
            review_ids.append(review_id)
            stars.append(star)

            if review_id == '#NAME?': review_id = None
            if business_id is None and review_id in review_business_id_dict: business_id = review_business_id_dict[review_id] # get business_id from review_business_id_dict

        if business_id is None: business_id = row['business_id_csv'] # if all review_id not in review_business_id_dict, then business_id in csv is used
        for text, review_id, star in zip(texts, review_ids, stars):
            ref_raw_dfs.append({'business_id': business_id, 'text': text, 'review_id': review_id, 'summary': summary, 'stars': star})

    ref_raw_df = pd.DataFrame(ref_raw_dfs)
    ref_raw_df['tokens'] = ref_raw_df['text'].apply(get_tokens)

    ref_df = ref_raw_df.groupby('business_id').agg({
        'text': lambda text_list: ' </DOC> '.join(list(text_list)),
        'tokens': lambda token_idxs_list: [sent_idxs for token_idxs in list(token_idxs_list) for sent_idxs in token_idxs],
        'summary': lambda summary_series: list(summary_series)[0], # only first column for each business id
        'stars': lambda stars_list: list(stars_list)
    })
    ref_df = ref_df.reset_index()
    ref_df['doc_l'] = ref_df['tokens'].apply(lambda tokens: len(tokens))
    ref_df['max_sent_l'] = ref_df['tokens'].apply(lambda tokens: max([len(line) for line in tokens]))
    ref_df['sent_l'] = ref_df['tokens'].apply(lambda tokens: [len(line) for line in tokens])
    ref_df['summary_tokens'] = ref_df['summary'].apply(get_tokens)
    ref_df['summary_doc_l'] = ref_df['summary_tokens'].apply(lambda tokens: len(tokens))
    ref_df['summary_max_sent_l'] = ref_df['summary_tokens'].apply(lambda tokens: max([len(line) for line in tokens]))
    return ref_df

def get_yelp_raw_df(data_df, ref_df):
    raw_df = data_df.groupby('business_id').agg({
        'tokens': lambda tokens_list: list(tokens_list),
        'stars': lambda stars_list: list(stars_list)
    })
    raw_df = raw_df.reset_index()
    ref_business_ids = ref_df['business_id'].values
    n_pre = len(raw_df)
    raw_df = raw_df[raw_df['business_id'].apply(lambda business_id: business_id not in ref_business_ids)]
    n_post = len(raw_df)
    print('filtered %i business ids in references from train datasets' % (n_pre-n_post))
    return raw_df

def get_yelp_df(config):
    get_data_paths = lambda data_dir: [os.path.join(data_dir, data_name) for data_name in os.listdir(data_dir) if not data_name == 'store-to-nreviews.json']
    train_data_paths = get_data_paths(config.dir_train)
    dev_data_paths = get_data_paths(config.dir_val)
    test_data_paths = get_data_paths(config.dir_test)

    train_tmp_df = apply_parallel(train_data_paths, n_processes=config.n_processes, map_func=get_yelp_tmp_df).reset_index()
    dev_tmp_df = apply_parallel(dev_data_paths, n_processes=config.n_processes, map_func=get_yelp_tmp_df).reset_index()
    test_tmp_df = apply_parallel(test_data_paths, n_processes=config.n_processes, map_func=get_yelp_tmp_df).reset_index()

    ref_df = get_yelp_ref_df(config.path_ref, train_tmp_df, dev_tmp_df, test_tmp_df)
    assert len(ref_df) == 200

    train_raw_df = get_yelp_raw_df(train_tmp_df, ref_df)
    train_df = get_group_df(train_raw_df, n_reviews=config.n_reviews, filter_sent_l=config.filter_sent_l, filter_doc_l=config.filter_doc_l, \
                        item_min_reviews=config.item_min_reviews, n_per_item=config.n_per_item)

    test_indices = [111,  74, 106,  45, 143, 147,  56, 150, 184,  38,  61, 125, 116,
                                 58, 159, 182,  86,   2,  22, 126,  55,  20, 161, 118, 141,  57,
                                123,  68, 164,  14, 122,  64,  53,  85, 135,  79, 163, 198, 109,
                                110,  25, 115, 113, 114,  78,  94, 151,  88, 162, 176,  66, 136,
                                 62, 137, 158, 148, 171, 145,  52,   1,  82,   5, 173, 124, 190,
                                129, 185,  67, 107,   3, 193, 132,  69,  31,  41,  11, 108, 167,
                                 96,  12, 139,  90,  23,  95,  21,   7,  54, 174,  65,  47, 194,
                                181, 153, 199, 121, 165,  80,  44, 188,  36]

    test_df = ref_df.iloc[test_indices]
    dev_df = ref_df.drop(test_df.index)
    assert len(set(test_df.index) & set(dev_df.index)) == 0

    return train_df, dev_df, test_df

def apply_token_idxs(tokens_series):
    def get_token_idxs(tokens):
        return [[word_to_idx[token] if token in word_to_idx else word_to_idx[UNK] for token in sent] for sent in tokens]
    return tokens_series.apply(get_token_idxs)


# -

if __name__ == '__main__':
    train_df, dev_df, test_df = get_yelp_df(config)
    
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
