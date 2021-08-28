#config: utf-8
import os
import sys
import argparse
import numpy as np
import _pickle as cPickle
import pdb

from recursum import RecursiveSummarizationModel
from tree import update_config_tree

def get_config(arg=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', type=str, default='0')
    parser.add_argument('-data', type=str, default=None)
    parser.add_argument('-model', type=str, default='recursum')
    parser.add_argument('-seed', type=int, default=1234)
    
    # dimenssion
    parser.add_argument('-emb', '--dim_emb', type=int, default=200)
    parser.add_argument('-hid', '--dim_hidden', type=int, default=400)
    parser.add_argument('-lat', '--dim_latent', type=int, default=32)

    # topic model
    parser.add_argument('-tree', type=str, default=None)
    parser.add_argument('-cov_root', type=float, default=1.)
    parser.add_argument('-minlv', '--min_logvar', type=float, default=0.)

    # encoder
    parser.add_argument('-bi', '--bidirectional', type=bool, default=True)
    parser.add_argument('-att', '--attention', type=bool, default=True)
    parser.add_argument('-cov', '--coverage', action='store_true')
    parser.add_argument('-oov', action='store_true')

    # decoder
    parser.add_argument('-nucleus', type=float, default=0.4)
    parser.add_argument('-beam', '--beam_width', type=int, default=5)
    parser.add_argument('-lpw', '--length_penalty_weight', type=float, default=0.)
    parser.add_argument('-parallel', '--parallel_iterations', type=int, default=32)
    parser.add_argument('-swap', '--swap_memory', type=bool, default=False)
    
    # gumbel softmax
    parser.add_argument('-maxt', '--max_temperature', type=float, default=1.)
    parser.add_argument('-mint', '--min_temperature', type=float, default=0.5)
    
    # learning configure
    parser.add_argument('-epoch', '--n_epochs', type=int, default=5)
    parser.add_argument('-step', '--n_steps', type=int, default=100000)
    parser.add_argument('-opt', default='adam')
    parser.add_argument('-batch', '--batch_size', type=int, default=8)
    parser.add_argument('-batch_eval', '--batch_eval_size', type=int, default=4)
    parser.add_argument('-lr', type=float, default=0.0005)
    parser.add_argument('-lr_disc', type=float, default=0.00005)
    parser.add_argument('-lr_step', action='store_true')
    parser.add_argument('-warmup', type=float, default=20000)
    parser.add_argument('-sample', action='store_true')
    parser.add_argument('-gl', '--grad_clip', type=float, default=5.)
    parser.add_argument('-wkp', '--word_keep_prob', type=float, default=0.75)
    parser.add_argument('-ekp', '--enc_keep_prob', type=float, default=0.8)
    parser.add_argument('-dkp', '--dec_keep_prob', type=float, default=0.8)

    # KL annealing
    parser.add_argument('-anneal', type=str, default='linear') # linear, cycle or constant    
    parser.add_argument('-beta_init', type=float, default=0.) # initial value of beta    
    parser.add_argument('-beta', '--beta_last', type=float, default=1.) # final value of beta    
    parser.add_argument('-linear', '--linear_steps', type=int, default=40000) # number of epochs within a cycle    
    parser.add_argument('-c_prob', '--capacity_prob', type=float, default=0.) # minimum kl
    parser.add_argument('-c_gmm', '--capacity_gmm', type=float, default=0.) # minimum kl
    parser.add_argument('-lam_disc', type=float, default=1.) # initial value of beta

    # sentence extraction
    parser.add_argument('-topk', type=int, default=8)
    parser.add_argument('-topk_train', type=int, default=4)
    parser.add_argument('-threshold', type=float, default=0.6)
    parser.add_argument('-suml', '--summary_l', type=int, default=6)
    parser.add_argument('-n_processes', type=int, default=16)

    # log
    parser.add_argument('-log', '--log_period', type=int, default=100)
    parser.add_argument('-max_to_keep', type=int, default=10)
    parser.add_argument('-dir_data', type=str, default='data')
    parser.add_argument('-dir_param', type=str, default='model')
    parser.add_argument('-name_data', type=str, default='data_df.pkl')
    parser.add_argument('-name_vocab', type=str, default='vocab.pkl')
    parser.add_argument('-name_eval', type=str, default='eval_df.pkl')
    parser.add_argument('-i_checkpoint', type=int, default=-1)
    parser.add_argument('-stable', action='store_true')
    
    # configure
    args = sys.argv[1:] if arg is None else arg.rstrip().split()
    config = parser.parse_args(args=args)
    
    # Model Class
    config.Model = RecursiveSummarizationModel
    
    # tree
    config = update_config_tree(config)
    
    # paths
    args_index = lambda s: args.index(s) if s in args else -100
    i_gpu = args_index('-gpu')
    i_data = args_index('-data')
    i_dir = args_index('-dir_data')
    i_processes = args_index('-n_processes')
    config.name_model = config.model + ''.join([arg for i, arg in enumerate(args) if i not in [i_gpu, i_gpu+1, i_data, i_data+1, i_dir, i_dir+1, i_processes, i_processes+1]])
    config.path_data = os.path.join(config.dir_data, config.data, config.name_data)
    config.path_vocab = os.path.join(config.dir_data, config.data, config.name_vocab)
    config.dir_model = os.path.join(config.dir_param, config.data, config.name_model)
    config.path_model = os.path.join(config.dir_model, 'model')
    config.path_eval = os.path.join(config.dir_model, config.name_eval)
    config.path_log = os.path.join(config.dir_model, 'log')
    config.path_txt = os.path.join(config.dir_model, 'txt')
    config.rouges_max = [0., 0., 0.]
    
    # dummy tokens
    config.PAD = '<pad>' # This has a vocab id, which is used to pad the encoder input, decoder input and target sequence
    config.UNK = '<unk>' # This has a vocab id, which is used to represent out-of-vocabulary words
    config.BOS = '<p>' # This has a vocab id, which is used at the beginning of every decoder input sequence
    config.EOS = '</p>' # This has a vocab id, which is used at the end of untruncated target sequences
    config.CLIP = 1e-8
    
    return config


def update_config(config, train_batches, dev_batches, word_to_idx):
    config.n_train = sum([len(batch) for batch in train_batches])
    config.n_vocab = len(word_to_idx)
    config.maximum_iterations = max([max([instance.summary_max_sent_l for instance in batch]) for batch in dev_batches]) + 1
    config.word_to_idx = word_to_idx
    config.idx_to_word = {idx: word for word, idx in word_to_idx.items()}

    config.PAD_IDX = word_to_idx[config.PAD]
    config.UNK_IDX = word_to_idx[config.UNK]
    config.BOS_IDX = word_to_idx[config.BOS]
    config.EOS_IDX = word_to_idx[config.EOS]
    return config
