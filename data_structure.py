#coding:utf-8
import os
import sys
import math
import random
import itertools
from itertools import zip_longest
import _pickle as cPickle
import numpy as np
from collections import defaultdict

class Instance:
    def __init__(self):
        self.token_idxs = None
        self.goldLabel = -1
        self.idx = -1

    def _doc_len(self, idx):
        k = len(self.token_idxs)
        return k

    def _max_sent_len(self, idxs):
        k = max([len(sent) for sent in self.token_idxs])
        return k

def get_batches(data_df, batch_size):
    instances = [instance for _, instance in data_df.iterrows()]
    batches = list(zip_longest(*[iter(instances)]*batch_size))
    batches = [tuple([instance for instance in batch if instance is not None]) for batch in batches]
    return batches

def get_batches_iterator(batches, log_period):
    batches = random.sample(batches, len(batches))
    batches_list = list(zip(*[iter(batches)]*log_period))
    return iter(batches_list)
