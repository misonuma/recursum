# +
import os
import pdb
import random
import _pickle as cPickle
from itertools import takewhile
from tqdm import tqdm

import numpy as np

from configure import get_config, update_config
from run import get_batches, get_batches_iterator, load, init, train
# -

# load config
config = get_config()

# load data
train_df, dev_df, _ = cPickle.load(open(config.path_data, 'rb'))
word_to_idx = cPickle.load(open(config.path_vocab, 'rb'))
train_batches = get_batches(train_df, config.batch_size)
dev_batches = get_batches(dev_df, config.batch_eval_size)

config = update_config(config, train_batches, dev_batches, word_to_idx)

# set gpu and seed
os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu
np.random.seed(config.seed)
random.seed(config.seed)

# buiild model
sess, model, saver = load(config)

# initialize log
epoch, log_df, logger = init(config)
train_batches_iterator = get_batches_iterator(train_batches, config.log_period)

# run model
epoch_iterator = iter(range(epoch, config.n_epochs))
for epoch in takewhile(lambda epoch: epoch is not None, epoch_iterator):
    for log_batches in takewhile(lambda log_batches: log_batches is not None, train_batches_iterator):
        sess, model, saver, log_df = train(sess, model, saver, log_batches, dev_df, log_df, logger, jupyter=False)
        if np.max(log_df.index) > config.n_steps: break
    train_batches_iterator = get_batches_iterator(train_batches, config.log_period)
    if np.max(log_df.index) > config.n_steps: break
epoch+=1


