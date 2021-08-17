import os
import sys
import pdb
import random
import _pickle as cPickle
import time
from itertools import takewhile
from matplotlib import pyplot as plt
import warnings
warnings.simplefilter('ignore', DeprecationWarning)
from tqdm import tqdm

import numpy as np
import pandas as pd
import tensorflow as tf
import texar.tf as tx

from configure import get_config, update_config
from data_structure import get_batches, get_batches_iterator
from run import load, init, train, print_summary, print_sample, idxs_to_sents

# load config
config = get_config()

# load data
train_df, dev_df, test_df, word_to_idx, idx_to_word, bow_idxs = cPickle.load(open(config.path_data, 'rb'))
train_batches = get_batches(train_df, config.batch_size)
dev_batches = get_batches(dev_df, config.batch_size)
test_batches = get_batches(test_df, config.batch_size)
config = update_config(config, train_batches, dev_batches, word_to_idx, bow_idxs)

# set gpu and seed
os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu
np.random.seed(config.seed)
random.seed(config.seed)

# buiild model
sess, model, saver = load(config)

# initialize log
epoch, log_df, logger = init(config, trash=False)
train_batches_iterator = get_batches_iterator(train_batches, config.log_period)

# run model
epoch_iterator = iter(range(epoch, config.n_epochs))
for epoch in takewhile(lambda epoch: epoch is not None, epoch_iterator):
    for log_batches in takewhile(lambda log_batches: log_batches is not None, train_batches_iterator):
        sess, model, saver, log_df, nan_flg = train(sess, model, saver, log_batches, dev_batches, test_batches, log_df, logger, jupyter=False)
        if np.max(log_df.index) > config.n_steps: break
    train_batches_iterator = get_batches_iterator(train_batches, config.log_period)
    if np.max(log_df.index) > config.n_steps: break
epoch+=1
