#!/usr/bin/env python
# coding: utf-8
# %%
import os
import sys
import pdb
import itertools
import _pickle as cPickle
import json
import argparse
import math
import random
from collections import OrderedDict, defaultdict, Counter
import re
import itertools
import multiprocessing

import nltk
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from configure import get_config, update_config
from data_structure import get_batches, get_batches_iterator
from run import init, train, print_summary, print_sample, idxs_to_sents, get_text_from_sents
from evaluation.eval_utils import EvalMetrics
from evaluate import get_evaluation_config, load_model, restore_model, evaluate_rouges, approx_rand_test, print_recursum
from evaluate import get_recursum_df, get_copycat_df, get_denoisesum_df, get_meansum_df, get_lead_df, get_lexrank_df, get_opinosis_df
from summarize import treesum, greedysum


# %%
def apply_parallel(datas, num_split, map_func):
    p = multiprocessing.Pool(processes=num_split)
    data_split = np.array_split(datas, num_split)
    output_dfs = p.map(map_func, data_split)
    p.close()
    output_df = pd.concat(output_dfs)
    return output_df


# %%
config = get_config()


# %%
np.random.seed(config.seed)
random.seed(config.seed)
eval_metrics = EvalMetrics()


# %%
n_path = -1
recursum_df = pd.read_pickle(config.path_recursum_df%n_path)


# %%
def get_perm_recursum_df(recursum_df):
    perm_recursum_dfs = []
    for test_index, row in recursum_df.iterrows():
        topic_sents = np.array(row.topic_sents)
        ref_summary = row.summary
        input_text = row.text.replace(' </DOC>', '').replace('\n', '')
        
        perms = itertools.permutations(range(config.n_topic), summary_doc_l)
        for perm in perms:
            perm_rouge = {}
            perm_rouge['test_index'] = test_index

            topic_indices = np.array(perm)
            topic_idxs = np.array([config.topic_idxs[topic_index] for topic_index in topic_indices])
            perm_rouge['topic_indices'] = topic_indices
            perm_rouge['topic_idxs'] = topic_idxs
            
            sys_summary = ' '.join(topic_sents[topic_indices])
            rouges = eval_metrics.calc_rouges(source=ref_summary, summary=sys_summary)
            # rouge_name = rouge1, rouge2, rougeL
            perm_rouge.update({rouge_name: getattr(rouge_obj, 'fmeasure') for rouge_name, rouge_obj in rouges.items()})
            
            rouges_input = eval_metrics.calc_rouges(source=input_text, summary=sys_summary)
            perm_rouge.update({'%s_input_%s' % (rouge_name, obj_n): getattr(rouge_obj, obj_name) \
                                                    for rouge_name, rouge_obj in rouges_input.items() \
                                                    for obj_name, obj_n in zip(['precision', 'recall', 'fmeasure'], ['p', 'r', 'f'])})
            perm_recursum_dfs.append(perm_rouge)
        
        print('finished %i' % test_index)

    perm_recursum_df = pd.DataFrame(perm_recursum_dfs)
    return perm_recursum_df


# %%
summary_doc_l = 4
oracle_recursum_df = apply_parallel(recursum_df, num_split=50, map_func=get_perm_recursum_df).reset_index()
oracle_recursum_df.to_pickle(config.path_oracle_df%n_path)
