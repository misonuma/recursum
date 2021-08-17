#coding: utf-8
# +
# %load_ext autoreload
# %autoreload

import os
import json
import pdb
import multiprocessing
import argparse
from collections import defaultdict
import _pickle as cPickle
import random

import tensorflow as tf
import numpy as np
import pandas as pd

from configure import get_config, update_config
from run import load, restore, get_batches, compute_topic_sents_probs, get_text_from_sents, compute_rouge
from summarize import greedysum


# -

def get_eval_df(sess, model, test_df, syssum, topk, threshold, summary_l, num_split):    
    data_df = test_df.groupby('business_id').agg({
        'doc_l': lambda doc_l_series: doc_l_series.values[0],
        'sent_l': lambda sent_l_series: sent_l_series.values[0],
        'token_idxs': lambda token_idxs_series: token_idxs_series.values[0],
        'text': lambda text_series: text_series.values[0]
    })

    batches = get_batches(data_df, model.config.batch_size)
    topic_sents_list, probs_topic_list, topic_tokens_list = compute_topic_sents_probs(sess, model, batches, mode='eval', sample=False)
    text_list = [row.text.replace('\n', '') for _, row in data_df.iterrows()]

    args = [(model.config.tree_idxs, topic_sents, text, topk, threshold, summary_l) for topic_sents, text in zip(topic_sents_list, text_list)]
    pool = multiprocessing.Pool(processes=num_split)
    summary_l_sents_list = pool.map(syssum, args)
    pool.close()

    summary_list = [get_text_from_sents(summary_l_sents[summary_l]['sents']) for summary_l_sents in summary_l_sents_list]
    summary_idxs_list = [[model.config.topic_idxs[topic_index] for topic_index in summary_l_sents[summary_l]['indices']] for summary_l_sents in summary_l_sents_list]

    data_df['recursum'] = summary_list
    data_df['summary_idxs'] = summary_idxs_list
    data_df['topic_sents'] = topic_sents_list
    data_df['probs_topic'] = probs_topic_list

    eval_df = pd.merge(test_df, data_df[['recursum', 'summary_idxs', 'topic_sents', 'probs_topic']], on='business_id', how='left')
    eval_df[['rouge1', 'rouge2', 'rougeL']] = compute_rouge(eval_df, reference_list=list(eval_df.summary), summary_list=list(eval_df.recursum))
    eval_df = eval_df.set_index(test_df.index)
    assert eval_df['business_id'].to_dict() == test_df['business_id'].to_dict()
    return eval_df

# nb_name = '3 yelp atttglm -tree 44 -sent -disc -turn -mean -prior -drnn -linear 40000 -lr 0.0005 -lr_disc 0.00005 -nucleus 0.4'
nb_name = '3 yelp atttglm -tree 44 -turn -linear 40000 -lr 0.0005 -lr_disc 0.00005 -nucleus 0.4'
n_path = -1

# load config
config = get_config(nb_name)

config.dir_model

# load data
train_df, dev_df, test_df, word_to_idx, idx_to_word, bow_idxs = cPickle.load(open(config.path_data, 'rb'))
train_batches = get_batches(train_df, config.batch_size)
dev_batches = get_batches(dev_df, config.batch_size)
test_batches = get_batches(test_df, config.batch_size)

config = update_config(config, train_batches, dev_batches, word_to_idx, bow_idxs)

# set gpu and seed
np.random.seed(config.seed)
random.seed(config.seed)

# buiild model
sess, model, saver = load(config)

sess = restore(sess, model, n_path)

# evaluate model
eval_df = get_eval_df(sess, model, test_df, syssum=greedysum, topk=8, threshold=0.6, summary_l=6, num_split=50)
eval_df[['rouge1', 'rouge2', 'rougeL']].mean()


