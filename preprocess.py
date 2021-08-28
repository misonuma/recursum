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

# +
# special tokens
PAD = '<pad>' # This has a vocab id, which is used to pad the encoder input, decoder input and target sequence
UNK = '<unk>' # This has a vocab id, which is used to represent out-of-vocabulary words
BOS = '<p>' # This has a vocab id, which is used at the beginning of every decoder input sequence
EOS = '</p>' # This has a vocab id, which is used at the end of untruncated target sequences
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

parser.add_argument('-data', type=str, default=None)

config = parser.parse_args()

if config.data == 'yelp':
    parser.add_argument('-n_per_item', type=int, default=12)
    parser.add_argument('-dir_train', type=str, default='data/yelp/train')
    parser.add_argument('-dir_val', type=str, default='data/yelp/val')
    parser.add_argument('-dir_test', type=str, default='data/yelp/test')
    parser.add_argument('-path_ref', type=str, default='data/yelp/references.csv')
elif config.data == 'amazon':
    parser.add_argument('-n_per_item', type=int, default=2)
    parser.add_argument('-dir_train', type=str, default='data/amazon/train')
    parser.add_argument('-path_dev', type=str, default='data/amazon/dev.csv')
    parser.add_argument('-path_test', type=str, default='data/amazon/test.csv')
else:
    raise
    
parser.add_argument('-path_output', type=str, default=os.path.join('data', config.data, 'data_df.pkl'))
parser.add_argument('-path_vocab', type=str, default=os.path.join('data', config.data, 'vocab.pkl'))
config = parser.parse_args()


# -

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


def apply_token_idxs(tokens_series):
    def get_token_idxs(tokens):
        return [[word_to_idx[token] if token in word_to_idx else word_to_idx[UNK] for token in sent] for sent in tokens]
    return tokens_series.apply(get_token_idxs)


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


# # yelp

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


# -

# # amazon

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


# -

# # main

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
