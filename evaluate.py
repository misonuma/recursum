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
from run import load, restore, get_batches, get_eval_df
from summarize import greedysum
# -

# load config
config = get_config()

# +
# load data
train_df, dev_df, test_df = cPickle.load(open(config.path_data, 'rb'))
word_to_idx = cPickle.load(open(config.path_vocab, 'rb'))

train_batches = get_batches(train_df, config.batch_size)
dev_batches = get_batches(dev_df, config.batch_size)
test_batches = get_batches(test_df, config.batch_size)
# -

# set gpu and seed
config = update_config(config, train_batches, dev_batches, word_to_idx)
os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu
np.random.seed(config.seed)
random.seed(config.seed)

# buiild and restore model
sess, model, saver = load(config)
sess = restore(sess, model, config.i_checkpoint)

# evaluate model
eval_df = get_eval_df(sess, model, test_df, syssum=greedysum, topk=config.topk, threshold=config.threshold, summary_l=config.summary_l, n_processes=config.n_processes)
print(eval_df[['rouge1', 'rouge2', 'rougeL']].mean())
