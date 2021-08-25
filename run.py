# +
import subprocess
import time
import _pickle as cPickle
from collections import defaultdict
import pdb
import logging
from itertools import zip_longest
import multiprocessing
import random

import numpy as np
import tensorflow as tf
import pandas as pd

from evaluation.rouge_scorer import RougeScorer
from summarize import greedysum
from IPython.display import clear_output


# -

def get_batches(data_df, batch_size):
    instances = [instance for _, instance in data_df.iterrows()]
    batches = list(zip_longest(*[iter(instances)]*batch_size))
    batches = [tuple([instance for instance in batch if instance is not None]) for batch in batches]
    return batches


def get_batches_iterator(batches, log_period):
    batches = random.sample(batches, len(batches))
    batches_list = list(zip(*[iter(batches)]*log_period))
    return iter(batches_list)


# +
def get_text_from_sents(sents):
    return ' '.join(sents)

def idxs_to_sents(token_idxs, config):
    sents = []
    for sent_idxs in token_idxs:
        sent = ''
        for index, idx in enumerate(sent_idxs):
            if idx == config.EOS_IDX: break
            
            word = config.idx_to_word[idx]
            
            if index==0 or word == 'i': word = word.capitalize()
            if index==0 or "'" in word or word == ',':
                sent += word
            else:
                sent += ' ' + word
        
        sent += '.'
        sents.append(sent)
    return sents


# +
def load(config):
    if 'sess' in globals(): sess.close()
    model = config.Model(config)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep=config.max_to_keep)
    return sess, model, saver

def restore(sess, model, n_path):
    ckpt = tf.train.get_checkpoint_state(model.config.dir_model)
    all_model_paths = ckpt.all_model_checkpoint_paths
    model_path = all_model_paths[n_path]
    
    saver = tf.train.Saver(max_to_keep=model.config.max_to_keep)
    saver.restore(sess, model_path)
    return sess


# -

def init(config):
    epoch = 0
    log_df = pd.DataFrame(columns=pd.MultiIndex.from_tuples(
        list(zip(*[['','',
                    'TRAIN','','','','','','',
                    'VALID','','','','','','','','','','','','','','',''],
                    ['TIME','BETA',
                     'LVAE','LDISC','RECON','PRKL','SEKL','DISE','DITO',\
                     'LVAE','LDISC','RECON','PRKL','SEKL','DISE','DITO','TOPIC','R1','R2','RL']]))))
    
    cmd_rm = 'rm -r %s' % config.dir_model
    res = subprocess.call(cmd_rm.split())
    cmd_mk = 'mkdir %s' % config.dir_model
    res = subprocess.call(cmd_mk.split())
    
    # logger
    logger = logging.getLogger(config.name_model)
    logger.setLevel(10)
    sh = logging.StreamHandler()
    logger.addHandler(sh)
    fh = logging.FileHandler(config.path_txt)
    logger.addHandler(fh)
    formatter = logging.Formatter('%(asctime)s:%(lineno)d:%(levelname)s:%(message)s')
    fh.setFormatter(formatter)
    sh.setFormatter(formatter)
    
    return epoch, log_df, logger


def train(sess, model, saver, train_batches, dev_df, log_df, logger, jupyter=True):
    time_start = time.time()

    # train
    losses_train, global_step = compute_loss(sess, model, train_batches, mode='train')
    log_train = ['%.2f'%loss for loss in losses_train]

    # dev
    dev_batches = get_batches(dev_df, model.config.batch_eval_size)
    losses_dev, _ = compute_loss(sess, model, dev_batches, mode='eval')
    log_dev = ['%.2f' % np.minimum(loss, 1e+4) for loss in losses_dev]
        
    eval_df = get_eval_df(sess, model, dev_df, syssum=greedysum, topk=model.config.topk_train, threshold=model.config.threshold, summary_l=model.config.summary_l, num_split=model.config.num_split)
    rouges = eval_df[['rouge1', 'rouge2', 'rougeL']].mean()
    log_rouges = ['%.3f'%rouge for rouge in rouges]
    
    beta = sess.run(model.beta)
    log_beta = '%.3f'%beta
    
    # save model
    if np.any([rouge > rouge_max for rouge, rouge_max in zip(rouges, model.config.rouges_max)]):
        model.config.rouges_max = rouges
        saver.save(sess, model.config.path_model, global_step=global_step)
    
    # print log
    if jupyter: clear_output()
    log_time = int(time.time() - time_start)
    time_start = time.time()
    log_df.loc[global_step] = pd.Series([log_time, log_beta] + log_train + log_dev + log_rouges, index=log_df.columns)
    if jupyter: display(log_df)
    log_df.to_pickle(model.config.path_log)
        
    # print sent
    dev_instance = dev_df.iloc[0]
    logger.info('######################### Step: %i #########################'%global_step)
    print_summary(dev_instance, sess, model, logger=logger)
    print_sample(dev_instance, sess, model, logger=logger)
        
    return sess, model, saver, log_df


def get_eval_df(sess, model, data_df, syssum, topk, threshold, summary_l, num_split):
    def compute_rouge(group_df, reference_list, summary_list):
        rouge_scorer = RougeScorer(rouge_types=['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        assert len(group_df) == len(reference_list) == len(summary_list)
        rouge_dict = {index: {rouge_name: getattr(rouge_obj, 'fmeasure') \
                                  for rouge_name, rouge_obj in rouge_scorer.score(target=reference, prediction=summary).items()}\
                                  for index, reference, summary in zip(group_df.index, reference_list, summary_list)}
        rouge_df = pd.DataFrame.from_dict(rouge_dict, orient='index')
        return rouge_df
    
    group_df = data_df.groupby('business_id').agg({
        'doc_l': lambda doc_l_series: doc_l_series.values[0],
        'sent_l': lambda sent_l_series: sent_l_series.values[0],
        'token_idxs': lambda token_idxs_series: token_idxs_series.values[0],
        'text': lambda text_series: text_series.values[0]
    })

    batches = get_batches(group_df, model.config.batch_eval_size)
    topic_sents_list, probs_topic_list = compute_topic_sents_probs(sess, model, batches, mode='eval')
    text_list = [row.text.replace('\n', '') for _, row in group_df.iterrows()]

    args = [(model.config.tree_idxs, topic_sents, text, topk, threshold, summary_l) for topic_sents, text in zip(topic_sents_list, text_list)]
    pool = multiprocessing.Pool(processes=num_split)
    summary_l_sents_list = pool.map(syssum, args)
    pool.close()

    summary_list = [' '.join(summary_l_sents[summary_l]['sents']) for summary_l_sents in summary_l_sents_list]
    summary_idxs_list = [[model.config.topic_idxs[topic_index] for topic_index in summary_l_sents[summary_l]['indices']] for summary_l_sents in summary_l_sents_list]

    group_df['recursum'] = summary_list
    group_df['summary_idxs'] = summary_idxs_list
    group_df['topic_sents'] = topic_sents_list
    group_df['probs_topic'] = probs_topic_list

    eval_df = pd.merge(data_df, group_df[['recursum', 'summary_idxs', 'topic_sents', 'probs_topic']], on='business_id', how='left')
    eval_df[['rouge1', 'rouge2', 'rougeL']] = compute_rouge(eval_df, reference_list=list(eval_df.summary), summary_list=list(eval_df.recursum))
    eval_df = eval_df.set_index(data_df.index)
    assert eval_df['business_id'].to_dict() == data_df['business_id'].to_dict()
    return eval_df


# +
def compute_loss(sess, model, batches, mode):
    losses = []
    for batch in batches:
        feed_dict = model.get_feed_dict(batch, mode=mode)
        if mode == 'train':
            try:
                _, _, global_step, loss_list = sess.run([model.opt, model.opt_disc, tf.train.get_global_step(), model.loss_list_train], feed_dict = feed_dict)
            except Exception as e:
                print(e) # some batch causes OOM
                continue
        elif mode == 'eval':
            global_step = None
            loss_list, = sess.run([model.loss_list_eval], feed_dict = feed_dict)
        losses += [loss_list]
    losses_mean = list(np.mean(losses, 0))
    return losses_mean, global_step

def compute_topic_sents_probs(sess, model, batches, mode):
    topic_sents_list, probs_topic_list = [], []
    
    for batch in batches:
        feed_dict = model.get_feed_dict(batch, mode=mode)
        
        batch_topic_token_idxs, batch_probs_topic = \
            sess.run([model.beam_summary_idxs, model.probs_topic_posterior], feed_dict=feed_dict)
        topic_sents_list += [idxs_to_sents(topic_token_idxs, model.config) for topic_token_idxs in batch_topic_token_idxs]
        probs_topic_list += list(batch_probs_topic)

    assert len(topic_sents_list) == len(probs_topic_list)
    return topic_sents_list, probs_topic_list

def print_summary(instance, sess, model, summary_sents=None, prob_topics=None, parent_idx=0, depth=0, sort=True, logger=None):
    def get_summary(instance, sess, model):
        feed_dict = model.get_feed_dict([instance], mode='eval')

        summary_idxs, probs_topics = sess.run([model.beam_summary_idxs, model.probs_sent_topic_posterior], feed_dict=feed_dict)

        summary_sents = idxs_to_sents(summary_idxs[0], model.config)
        prob_topics = np.sum(probs_topics[0], 0) / np.sum(probs_topics[0]) # n_topic
        return summary_sents, prob_topics
    
    if summary_sents is None and prob_topics is None:
        # compute summary sentences only once
        summary_sents, prob_topics = get_summary(instance, sess, model)

    if depth == 0:
        # print root
        logger.info('-----------Topic sentences-----------')
        if instance is not None: logger.info('Item idx: %s' % instance.business_id)
        pred_summary_sent = summary_sents[model.config.topic_idxs.index(parent_idx)]
        prob_topic = prob_topics[model.config.topic_idxs.index(parent_idx)]
        logger.info('%s P: %.3f SENT: %s' % (parent_idx, prob_topic, pred_summary_sent))

    child_idxs = model.config.tree_idxs[parent_idx]
    depth += 1
    for child_idx in child_idxs:
        # print children recursively
        pred_summary_sent = summary_sents[model.config.topic_idxs.index(child_idx)]
        prob_topic = prob_topics[model.config.topic_idxs.index(child_idx)]
        logger.info('  '*depth + '%i P: %.3f SENT: %s' % (child_idx, prob_topic, pred_summary_sent))
        if child_idx in model.config.tree_idxs:
            print_summary(instance, sess, model, summary_sents, prob_topics, parent_idx=child_idx, depth=depth, logger=logger)
            
def print_sample(test_instance, sess, model, n_sents=None, logger=None): 
    feed_dict = model.get_feed_dict([test_instance], mode='eval')
    pred_token_idxs = sess.run(model.beam_output_idxs, feed_dict = feed_dict)[0]
    pred_sents = idxs_to_sents(pred_token_idxs, model.config)
    true_sents = idxs_to_sents(test_instance.token_idxs, model.config)
    assert len(pred_sents) == len(true_sents)

    logger.info('-----------Reconstructed Sentences-----------')
    for i, (true_sent, pred_sent) in enumerate(zip(true_sents[:n_sents], pred_sents[:n_sents])):
        logger.info('%i TRUE: %s' % (i, true_sent))
        logger.info('%i PRED: %s' % (i, pred_sent))
