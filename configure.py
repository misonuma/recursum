#config: utf-8
import os
import sys
import argparse
import numpy as np
import _pickle as cPickle
import glob
import pdb
from collections import defaultdict
import re

from tree import get_topic_idxs, get_child_to_parent_idxs, get_depth, get_descendant_idxs, \
                        get_mask_tree_reg, get_mask_tree, get_mask_tree_sibling, get_mask_tree_other, get_mask_tree_reverse
from hiertglm import HierarchicalTopicGuidedLanguageModel
from atttglm import AttentionTopicGuidedLanguageModel

def get_config(nb_name=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('gpu')
    parser.add_argument('data')
    parser.add_argument('model')
    
    parser.add_argument('-seed', type=int, default=1234)
    parser.add_argument('-max_to_keep', type=int, default=32)
    
    # data
    parser.add_argument('-small', action='store_true')
    parser.add_argument('-large', action='store_true')
    
    # dimenssion
    parser.add_argument('-emb', '--dim_emb', type=int, default=200)
    parser.add_argument('-hid', '--dim_hidden', type=int, default=400)
    parser.add_argument('-lat', '--dim_latent', type=int, default=32)

    # topic model
    parser.add_argument('-tree', type=str, default=None)
    parser.add_argument('-topic', '--n_topic', type=int, default=21)
    parser.add_argument('-dep', '--n_depth', type=int, default=3)
    parser.add_argument('-tm', '--topic_model', action='store_true')
    parser.add_argument('-cell', type=str, default='gru')
    parser.add_argument('-temp', '--depth_temperature', type=float, default=1.)
    parser.add_argument('-drnn', '--renew_drnn', action='store_true')
    parser.add_argument('-dropout_drnn', action='store_true')
    parser.add_argument('-ln', '--layernorm_drnn', type=bool, default=True)
    
    parser.add_argument('-latent', action='store_true')
    parser.add_argument('-cov_root', type=float, default=1.)
    parser.add_argument('-cov_root_e', type=float, default=0.)
#     parser.add_argument('-maxlv', '--max_logvar', type=float, default=1.)
    parser.add_argument('-minlv', '--min_logvar', type=float, default=0.)
    parser.add_argument('-minlv_inf', '--min_logvar_inf', action='store_true')
    parser.add_argument('-minlv_e', '--min_logvar_e', type=float, default=None)
    parser.add_argument('-prior', action='store_true')
    
    parser.add_argument('-maxlv_bow', '--max_logvar_bow', type=float, default=np.inf)
    parser.add_argument('-minlv_bow', '--min_logvar_bow', type=float, default=-np.inf)
        
    # language model
    parser.add_argument('-bi', '--bidirectional', type=bool, default=True)
    parser.add_argument('-att', '--attention', type=bool, default=True)
    parser.add_argument('-nonatt', action='store_true')
    parser.add_argument('-pg', '--pointer', action='store_true')
    parser.add_argument('-cov', '--coverage', action='store_true')
    parser.add_argument('-oov', action='store_true')
    parser.add_argument('-reg', '--regularizer', action='store_true')
    
    parser.add_argument('-maxt', '--max_temperature', type=float, default=1.)
    parser.add_argument('-mint', '--min_temperature', type=float, default=0.5)
    
    parser.add_argument('-nucleus', type=float, default=1.)
    parser.add_argument('-beam', '--beam_width', type=int, default=5)
    parser.add_argument('-lpw', '--length_penalty_weight', type=float, default=0.)
    parser.add_argument('-parallel', '--parallel_iterations', type=int, default=32)
    parser.add_argument('-swap', '--swap_memory', type=bool, default=False)
    
    # learning configure
    parser.add_argument('-opt', default='adam')
    parser.add_argument('-lr', type=float, default=0.001)
    parser.add_argument('-lr_disc', type=float, default=0.001)
    parser.add_argument('-lr_step', action='store_true')
    parser.add_argument('-warmup', type=float, default=20000)
    parser.add_argument('-sample', action='store_true')
    parser.add_argument('-turn', '--turn', action='store_true')
    parser.add_argument('-control', action='store_true')
    parser.add_argument('-disc', '--disc_topic', action='store_true')
    parser.add_argument('-sent', '--disc_sent', action='store_true')    
    parser.add_argument('-mean', '--disc_mean', action='store_true')
    parser.add_argument('-gumbel', '--disc_gumbel', type=bool, default=True)
    parser.add_argument('-weight', '--disc_weight', action='store_true')

    parser.add_argument('-epoch', '--n_epochs', type=int, default=5)
    parser.add_argument('-step', '--n_steps', type=int, default=1000000)
    parser.add_argument('-save', type=str, default='rouge')
    parser.add_argument('-save_steps', type=int, default=15000)
    parser.add_argument('-batch', '--batch_size', type=int, default=8)
    parser.add_argument('-eval_batch', '--eval_batch_size', type=int, default=2)
    parser.add_argument('-log', '--log_period', type=int, default=500)
    parser.add_argument('-txt', '--txt_period', type=int, default=1000)
    parser.add_argument('-n_freq', type=int, default=10)
    parser.add_argument('-avg', action='store_true')
    parser.add_argument('-pretrain', type=str, default='')
    parser.add_argument('-load', action='store_true')
    
    parser.add_argument('-gl', '--grad_clip', type=float, default=5.)
    parser.add_argument('-wkp', '--word_keep_prob', type=float, default=0.75)
    parser.add_argument('-ekp', '--enc_keep_prob', type=float, default=0.8)
    parser.add_argument('-dkp', '--dec_keep_prob', type=float, default=0.8)    
    
    # KL annealing
    parser.add_argument('-anneal', type=str, default='linear') # linear, cycle or constant    
    parser.add_argument('-beta_init', type=float, default=0.) # initial value of beta    
    parser.add_argument('-beta', '--beta_last', type=float, default=1.) # final value of beta    
    parser.add_argument('-cycle', '--cycle_steps', type=int, default=20000) # number of epochs within a cycle
    parser.add_argument('-linear', '--linear_steps', type=int, default=80000) # number of epochs within a cycle    
    parser.add_argument('-rate', '--r_cycle', type=float, default=0.5) # proportion used to increase beta within a cycle
    parser.add_argument('-c_prob', '--capacity_prob', type=float, default=0.) # minimum kl
    parser.add_argument('-c_gmm', '--capacity_gmm', type=float, default=0.) # minimum kl
    parser.add_argument('-beta_disc', action='store_true') # initial value of beta
    parser.add_argument('-lam_disc', type=float, default=1.) # initial value of beta    
    parser.add_argument('-lam_reg', type=float, default=1.) # initial value of beta
    
    # aggressive training
    parser.add_argument('-agg', '--aggressive', action='store_true') # flg of aggressive
    parser.add_argument('-n_agg', '--n_aggressive', type=int, default=30) # number of iteration in aggressive training
    parser.add_argument('-n_sample', type=int, default=1) # number of sample for computing mutual information
    parser.add_argument('-mi', '--n_loss_mi', type=float, default=10) # thereshold number of mutual information convergence
    
    # reverse KL
    parser.add_argument('-r_kl', '--reverse_kl', action='store_true')
    parser.add_argument('-mask', '--mask_tree_type', type=str, default='tree')
    parser.add_argument('-margin', '--margin_gmm', type=float, default=0.)
    
    parser.add_argument('-tmp', action='store_true')
    parser.add_argument('-test', action='store_true')
    parser.add_argument('-rev', action='store_true')
    
    parser.add_argument('-r0', action='store_true')
    parser.add_argument('-r1', action='store_true')
    parser.add_argument('-r2', action='store_true')
    parser.add_argument('-r3', action='store_true')
    parser.add_argument('-r4', action='store_true')
    parser.add_argument('-r5', action='store_true')
    parser.add_argument('-r6', action='store_true')
    parser.add_argument('-r7', action='store_true')
    parser.add_argument('-r8', action='store_true')
    parser.add_argument('-r9', action='store_true')
    parser.add_argument('-r10', action='store_true')
    parser.add_argument('-r11', action='store_true')
    parser.add_argument('-r12', action='store_true')
    parser.add_argument('-r13', action='store_true')
    parser.add_argument('-r14', action='store_true')
    parser.add_argument('-r15', action='store_true')
    parser.add_argument('-r16', action='store_true')
    parser.add_argument('-r17', action='store_true')
    parser.add_argument('-r18', action='store_true')
    parser.add_argument('-r19', action='store_true')
    parser.add_argument('-r20', action='store_true')
    parser.add_argument('-r21', action='store_true')
    parser.add_argument('-r22', action='store_true')
    parser.add_argument('-r23', action='store_true')
    parser.add_argument('-r24', action='store_true')
    parser.add_argument('-r25', action='store_true')
    parser.add_argument('-r26', action='store_true')
    parser.add_argument('-r27', action='store_true')
    parser.add_argument('-r28', action='store_true')
    parser.add_argument('-r29', action='store_true')
    parser.add_argument('-r30', action='store_true')
    parser.add_argument('-r31', action='store_true')

    # configure
    args = nb_name.replace('.ipynb', '').rstrip().split() if nb_name is not None else sys.argv[1:]
    config = parser.parse_args(args=args)
    
    if config.nonatt:
        config.attention = False
    
    if config.min_logvar_inf:
        config.min_logvar = -np.inf
    
    if config.cov_root_e != 0.:
        config.cov_root = np.e ** config.cov_root_e
    
    # Model Class
    if config.model == 'hiertglm':
        config.Model = HierarchicalTopicGuidedLanguageModel
    elif config.model == 'atttglm':
        config.Model = AttentionTopicGuidedLanguageModel
    
    # tree
    if config.tree is not None:
        tree_idxs = get_tree_idxs(config.tree)
        config = update_config_tree(config, tree_idxs)
    else:
        config.topic_idxs = [i for i in range(config.n_topic)]
    
    # paths
    config.fname_model = ''.join(args[2:])
    config.path_data = os.path.join('data', config.data, '%s_df.pkl' % config.data)
    if config.oov: config.path_data = os.path.join('data', config.data, '%s_oov_df.pkl' % config.data)
    
    config.dir_model = os.path.join('model', config.data, config.model, config.fname_model)
    config.dir_corpus = os.path.join('corpus', config.data) 
    config.path_model = os.path.join(config.dir_model, 'model')
    config.path_recursum_df = os.path.join(config.dir_model, 'recursum_df%i.pkl')
    config.path_oracle_df = os.path.join(config.dir_model, 'oracle_df%i.pkl')
    config.path_log = os.path.join(config.dir_model, 'log')
    config.path_txt = os.path.join(config.dir_model, 'txt')
    config.path_timeline = os.path.join(config.dir_model, 'timeline.json')
    config.path_checkpoint = os.path.join(config.dir_model, 'checkpoint')
    config.dir_eval = os.path.join('eval', config.data)
    config.dir_pretrain = os.path.join('model', config.data, config.model, 'treetglm-tree%s-pretraintm-lr5e-4'%config.tree)
    
    # hyperparameters for aggressive training
    config.ppl_min = np.inf
    config.rouges_max = [0., 0., 0.]
    
    config.max_loss_mi = 0.
    config.i_loss_mi = 0
    
    # dummy tokens
    config.PAD = '<pad>' # This has a vocab id, which is used to pad the encoder input, decoder input and target sequence
    config.UNK = '<unk>' # This has a vocab id, which is used to represent out-of-vocabulary words
    config.BOS = '<p>' # This has a vocab id, which is used at the beginning of every decoder input sequence
    config.EOS = '</p>' # This has a vocab id, which is used at the end of untruncated target sequences
    config.CLIP = 1e-8
        
    return config





def get_tree_idxs(tree, tree_idxs=None, depth=0, parent_idx=0):
    if tree_idxs is None: tree_idxs = {}
    child_idxs = [parent_idx*10+i+1 for i in range(int(tree[depth]))]
    tree_idxs[parent_idx] = child_idxs
    if depth+1 < len(tree):
        for child_idx in child_idxs:
            tree_idxs = get_tree_idxs(tree, tree_idxs=tree_idxs, depth=depth+1, parent_idx=child_idx)
    return tree_idxs

# +
def update_config_tree(config, tree_idxs):
    config.tree_idxs = tree_idxs
    config.topic_idxs = get_topic_idxs(config.tree_idxs)
    config.n_topic=sum([len(child_idxs) for child_idxs in config.tree_idxs.values()]) + 1
    
    config.child_to_parent_idxs = get_child_to_parent_idxs(config.tree_idxs)
    config.all_child_idxs = list(config.child_to_parent_idxs.keys()) # n_topic - 1
    
    config.tree_depth = get_depth(config.tree_idxs)
    config.depth_topic_idxs = defaultdict(list)
    for topic_idx, depth in config.tree_depth.items():
        config.depth_topic_idxs[depth].append(topic_idx)
    config.n_depth = max(config.tree_depth.values())
   
    if config.prior:
        depth_probs_topic_prior = {depth: 1./config.n_depth/len(topic_idxs) for depth, topic_idxs in config.depth_topic_idxs.items()}
        config.probs_topic_prior = np.array([depth_probs_topic_prior[config.tree_depth[topic_idx]] for topic_idx in config.topic_idxs], dtype=np.float32)
    
#     config.mask_tree_reg = get_mask_tree_reg(config.tree_idxs, config.all_child_idxs)
#     if config.mask_tree_type == 'tree':
#         config.mask_tree = get_mask_tree(config.tree_idxs, config.topic_idxs, config.depth_topic_idxs)
#     elif config.mask_tree_type == 'sibling':
#         config.mask_tree = get_mask_tree_sibling(config.tree_idxs, config.topic_idxs, config.depth_topic_idxs)
#     elif config.mask_tree_type == 'other':
#         config.mask_tree = get_mask_tree_other(config.tree_idxs, config.topic_idxs)
#     elif config.mask_tree_type == 'reverse':
#         config.mask_tree = get_mask_tree_reverse(config.tree_idxs, config.topic_idxs, config.depth_topic_idxs)        
#     else:
#         raise
    return config


# -

def update_config(config, train_batches, dev_batches, word_to_idx, bow_idxs):
    config.n_train = sum([len(batch) for batch in train_batches])
    config.n_vocab = len(word_to_idx)
    config.dim_bow = len(bow_idxs) if bow_idxs is not None else 0
    config.maximum_iterations = max([max([instance.summary_max_sent_l for instance in batch]) for batch in dev_batches]) + 1
#     config.cycle_steps = len(train_batches)*config.epochs_cycle # number of steps for each cycle
    config.word_to_idx = word_to_idx
    config.idx_to_word = {idx: word for word, idx in word_to_idx.items()}
    config.bow_idxs = bow_idxs

    config.PAD_IDX = word_to_idx[config.PAD]
    config.UNK_IDX = word_to_idx[config.UNK]
    config.BOS_IDX = word_to_idx[config.BOS]
    config.EOS_IDX = word_to_idx[config.EOS]
    return config

def update_checkpoint(config, checkpoint, global_step):
    checkpoint.append('model-%i' % global_step)
    if len(checkpoint) > config.max_to_keep:
        path_model = checkpoint.pop(0) + '.*'
        for p in glob.glob(path_model):
            os.remove(p)
    checkpoint_lines = ['model_checkpoint_path: "%s"' % checkpoint[-1]]
    checkpoint_lines += ['all_model_checkpoint_paths: "%s"' % model for model in checkpoint]
    
    with open(config.path_checkpoint, 'w') as f:
        f.writelines('\n'.join(checkpoint_lines))


def eval_config(nb_name_cw):
    sh_name, num_gpu = nb_name_cw.replace('.py', '').split('-')
    
    with open('%s.sh'%sh_name) as f:
        script = f.read()
    nb_name_list = re.findall(r'python main.py (.*) &', script)

    for nb_name in nb_name_list:
        if nb_name.split()[0] == num_gpu:
            nb_name_cw = nb_name

    config = get_config(nb_name_cw)
    print(nb_name_cw)
    return config
