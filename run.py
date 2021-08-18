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
pd.set_option('display.max_columns', 50)

rouge_scorer = RougeScorer(rouge_types=['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)


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
def get_sents_from_tokens(tokens):
    return [' '.join(line_tokens) for line_xtaaokens in tokens]

def get_text_from_sents(sents):
    return ' '.join(sents)

def idxs_to_sents(token_idxs, config):
    sents = []
    for sent_idxs in token_idxs:
        sent = ''
        for index, idx in enumerate(sent_idxs):
            if idx == config.EOS_IDX: break
            
            if idx in config.idx_to_word:
                word = config.idx_to_word[idx]
            else:
                word = config.idx_to_oov[idx]
            
            if index==0 or word == 'i': word = word.capitalize()
            if index==0 or "'" in word or word == ',':
                sent += word
            else:
                sent += ' ' + word
        
        sent += '.'
        sents.append(sent)
    return sents

def idxs_to_tokens(token_idxs, config):
    tokens = []
    for sent_idxs in token_idxs:
        sent_tokens = []
        for index, idx in enumerate(sent_idxs):
            if idx == config.EOS_IDX: break
            
            if idx in config.idx_to_word:
                word = config.idx_to_word[idx]
            else:
                word = config.idx_to_oov[idx]
            
            sent_tokens.append(word)
        
        tokens.append(sent_tokens)
    return tokens

def tokens_to_sents(tokens, config):
    sents = []
    for sent_tokens in tokens:
        sent = ''
        for index, token in enumerate(sent_tokens):
            if index==0 or token == 'i': token = token.capitalize()
            if index==0 or "'" in token or token == ',':
                sent += token
            else:
                sent += ' ' + token
        sent += '.'
        sents.append(sent)
    return sents


# +
def load(config):
    if config.load:
        tf.reset_default_graph()
        ckpt = tf.train.get_checkpoint_state(config.dir_pretrain)
        model_path = ckpt.all_model_checkpoint_paths[-1]
        saver = tf.train.import_meta_graph('%s.meta' % model_path)
        if 'sess' in globals(): sess.close()
        sess = tf.Session()
        saver.restore(sess, model_path)
        name_variables = {tensor.name: variable for tensor, variable in zip(tf.trainable_variables(), sess.run(tf.trainable_variables()))} # store paremeters
    
    if 'sess' in globals(): sess.close()
    model = config.Model(config)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep=config.max_to_keep)
        
    if config.load:
        name_tensors = {tensor.name: tensor for tensor in tf.global_variables()}
        sess.run([name_tensors[name].assign(variable) for name, variable in name_variables.items()]) # restore parameters
    return sess, model, saver

def restore(sess, model, n_path):
    ckpt = tf.train.get_checkpoint_state(model.config.dir_model)
    all_model_paths = ckpt.all_model_checkpoint_paths
    model_path = all_model_paths[n_path]
    
    saver = tf.train.Saver(max_to_keep=model.config.max_to_keep)
    saver.restore(sess, model_path)
    return sess


# +
def init(config, trash=True):
    epoch = 0
    log_df = pd.DataFrame(columns=pd.MultiIndex.from_tuples(
        list(zip(*[['','','',
                    'TRAIN','','','','','','','',
                    'VALID','','','','','','','','','','','','','','','','',''],
                    ['TIME','BETA',
                     'PPL','LVAE','LDISC','RECON','PRKL','SEKL','DISE','DITO',\
                     'PPL','LVAE','LDISC','RECON','PRKL','SEKL','DISE','DITO','TOPIC','R1','R2','RL','LCOV','ERR']]))))
    
    cmd_rm = 'trash -r %s' % config.dir_model if trash else 'rm -r %s' % config.dir_model
    res = subprocess.call(cmd_rm.split())
    cmd_mk = 'mkdir %s' % config.dir_model
    res = subprocess.call(cmd_mk.split())
    
    # logger
    logger = logging.getLogger(config.fname_model)
    logger.setLevel(10)
    sh = logging.StreamHandler()
    logger.addHandler(sh)
    fh = logging.FileHandler(config.path_txt)
    logger.addHandler(fh)
    formatter = logging.Formatter('%(asctime)s:%(lineno)d:%(levelname)s:%(message)s')
    fh.setFormatter(formatter)
    sh.setFormatter(formatter)
    
    return epoch, log_df, logger

def train(sess, model, saver, train_batches, dev_df, log_df, logger, sample=False, debug=False, jupyter=True):
    time_start = time.time()

    # train
    ppl_losses_train, global_step, n_errors = compute_loss(sess, model, train_batches, mode='train', sample=True, debug=debug)

    log_train = ['%.2f' % np.minimum(loss, 1e+4) for loss in ppl_losses_train]
    log_errors = str(n_errors)
    
    loss_train = ppl_losses_train[1]
    if loss_train > 10**4:
        print('Nan occured')
        ckpt = tf.train.get_checkpoint_state(model.config.dir_model)
        model_checkpoint_path = ckpt.all_model_checkpoint_paths[-1]
        saver.restore(sess, model_checkpoint_path)
        nan_flg=True
        return sess, model, saver, log_df, nan_flg
    else:
        nan_flg=False

    # dev
    dev_batches = get_batches(dev_df, model.config.batch_size)
    ppl_losses_dev, _, _ = compute_loss(sess, model, dev_batches, mode='eval', sample=sample)
    log_dev = ['%.2f' % np.minimum(loss, 1e+4) for loss in ppl_losses_dev]
    
    depth_mean_logdetcovs_topic_posterior = compute_logdetcovs(sess, model, dev_batches, mode='eval', sample=sample)
    log_logdetcovs = ' | '.join(['%.1f' % logdetcov for logdetcov in depth_mean_logdetcovs_topic_posterior])
    ppl_dev = ppl_losses_dev[0]
    
    eval_df = get_eval_df(sess, model, dev_df, syssum=greedysum, topk=model.config.topk_train, threshold=model.config.threshold, summary_l=model.config.summary_l, num_split=model.config.num_split)
    rouges = eval_df[['rouge1', 'rouge2', 'rougeL']].mean()
    log_rouges = ['%.3f'%rouge for rouge in rouges]
    
    beta = sess.run(model.beta)
    log_beta = '%.3f'%beta
    
    # save model
    if global_step > model.config.save_steps:
        if model.config.save == 'ppl' and ppl_dev < model.config.ppl_min:
            model.config.ppl_min = ppl_dev
            saver.save(sess, model.config.path_model, global_step=global_step)
        elif model.config.save == 'rouge' and np.any([rouge > rouge_max for rouge, rouge_max in zip(rouges, model.config.rouges_max)]):
            model.config.rouges_max = rouges
            saver.save(sess, model.config.path_model, global_step=global_step)
    
    # print log
    if jupyter: clear_output()
    log_time = int(time.time() - time_start)
    time_start = time.time()
    try:
        log_df.loc[global_step] = pd.Series([log_time, log_beta] + log_train + log_dev + log_rouges + [log_logdetcovs] + [log_errors], index=log_df.columns)
    except Exception as e:
        print(e)
        pdb.set_trace()
    if jupyter: display(log_df)
    log_df.to_pickle(model.config.path_log)
        
    # print sent
    dev_instance = dev_df.iloc[0]
    logger.info('######################### Step: %i #########################'%global_step)
    print_summary(dev_instance, sess, model, beam=True, sample=False, logger=logger)
    print_sample(dev_instance, sess, model, logger=logger)
        
    return sess, model, saver, log_df, nan_flg


# -

def get_eval_df(sess, model, data_df, syssum, topk, threshold, summary_l, num_split):
    def compute_rouge(group_df, reference_list, summary_list):
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

    batches = get_batches(group_df, model.config.batch_size)
    topic_sents_list, probs_topic_list, topic_tokens_list = compute_topic_sents_probs(sess, model, batches, mode='eval', sample=False)
    text_list = [row.text.replace('\n', '') for _, row in group_df.iterrows()]

    args = [(model.config.tree_idxs, topic_sents, text, topk, threshold, summary_l) for topic_sents, text in zip(topic_sents_list, text_list)]
    pool = multiprocessing.Pool(processes=num_split)
    summary_l_sents_list = pool.map(syssum, args)
    pool.close()

    summary_list = [get_text_from_sents(summary_l_sents[summary_l]['sents']) for summary_l_sents in summary_l_sents_list]
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
def compute_loss(sess, model, batches, mode, sample, debug=False):
    def check(variable, sample_batch=None):
        if sample_batch is None: sample_batch = batches[0]
        feed_dict = model.get_feed_dict(sample_batch, mode='eval', assertion=debug)
        feed_dict[model.t_variables['sample']] = sample
        _variable = sess.run(variable, feed_dict=feed_dict)
        return _variable
    
    losses = []
    loss_ppl = 0
    n_tokens = 0
    n_errors = 0

    for batch in batches:
        feed_dict = model.get_feed_dict(batch, mode=mode)
        feed_dict[model.t_variables['sample']] = sample
        run_options = tf.RunOptions(report_tensor_allocations_upon_oom=debug)
        
        try:
            if mode == 'train':
                _, _, global_step, loss_list, loss_sum = \
                    sess.run([model.opt, model.opt_disc, tf.train.get_global_step(), model.loss_list_train, model.loss_sum], feed_dict = feed_dict, options=run_options)
            elif mode == 'eval':
                global_step = None
                loss_list, loss_sum = sess.run([model.loss_list_eval, model.loss_sum], feed_dict = feed_dict, options=run_options)
                    
        except tf.errors.InvalidArgumentError as ie:
            print(ie)
            pdb.set_trace()

        except Exception as e:
            print(e)
            n_errors += 1
            pdb.set_trace()
        
        losses += [loss_list]
        loss_ppl += loss_sum # for computing PPL
        n_tokens += np.sum(feed_dict[model.t_variables['dec_sent_l']]) # for computing PPL

    losses_mean = list(np.mean(losses, 0))
    ppl = np.exp((loss_ppl)/n_tokens)
    ppl_losses = [ppl] + losses_mean
    
    return ppl_losses, global_step, n_errors

def compute_logdetcovs(sess, model, batches, mode, sample):
    depth_logdetcovs_topic_posterior_dict = defaultdict(list)
    
    for batch in batches:
        feed_dict = model.get_feed_dict(batch, mode=mode)
        feed_dict[model.t_variables['sample']] = sample
        
        depth_logdetcovs_topic_posterior = sess.run(model.depth_logdetcovs_topic_posterior, feed_dict=feed_dict)
        
        for depth, logdetcovs_topic_posterior in enumerate(depth_logdetcovs_topic_posterior):
            depth_logdetcovs_topic_posterior_dict[depth] += [logdetcovs_topic_posterior]
            
    depth_mean_logdetcovs_topic_posterior = np.array([np.mean(np.concatenate(logdetcovs_topic_posterior_list)) \
      for depth, logdetcovs_topic_posterior_list in depth_logdetcovs_topic_posterior_dict.items()])
    
    return depth_mean_logdetcovs_topic_posterior

def compute_topic_sents_probs(sess, model, batches, mode, sample):
    topic_tokens_list, topic_sents_list, probs_topic_list = [], [], []
    
    for batch in batches:
        feed_dict = model.get_feed_dict(batch, mode=mode)
        feed_dict[model.t_variables['sample']] = sample
        
        batch_topic_token_idxs, batch_probs_topic = \
            sess.run([model.beam_summary_idxs, model.probs_topic_posterior], feed_dict=feed_dict)
        topic_tokens_list += [idxs_to_tokens(topic_token_idxs, model.config) for topic_token_idxs in batch_topic_token_idxs]
        topic_sents_list += [idxs_to_sents(topic_token_idxs, model.config) for topic_token_idxs in batch_topic_token_idxs]
        probs_topic_list += list(batch_probs_topic)

    assert len(topic_sents_list) == len(probs_topic_list) == len(topic_tokens_list)
    return topic_sents_list, probs_topic_list, topic_tokens_list

def print_sample(test_instance, sess, model, n_sents=None, console=True, logger=None):
    if logger is None:
        logger = logging.getLogger(model.config.fname_model)
        logger.setLevel(10)
        sh = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s:%(lineno)d:%(levelname)s:%(message)s')
        sh.setFormatter(formatter)
        logger.addHandler(sh)    
    
    sample_list = []
    feed_dict = model.get_feed_dict([test_instance], mode='eval')
    pred_token_idxs = sess.run(model.beam_output_idxs, feed_dict = feed_dict)[0]
    pred_sents = idxs_to_sents(pred_token_idxs, model.config)
    true_sents = tokens_to_sents(test_instance.tokens, model.config) if model.config.oov \
                                else idxs_to_sents(test_instance.token_idxs, model.config)
    assert len(pred_sents) == len(true_sents)

    sample_list.append('-----------Reconstructed Sentences-----------')
    for i, (true_sent, pred_sent) in enumerate(zip(true_sents[:n_sents], pred_sents[:n_sents])):
        logger.info('%i TRUE: %s' % (i, true_sent))
        logger.info('%i PRED: %s' % (i, pred_sent))
        
    return sample_list

def get_summary(instance, sess, model, beam=True, sample=True):
    feed_dict = model.get_feed_dict([instance], mode='eval')
    feed_dict[model.t_variables['sample']] = sample

    if beam:
        summary_idxs, probs_topics = sess.run([model.beam_summary_idxs, model.probs_sent_topic_posterior], feed_dict=feed_dict)
    else:
        summary_idxs, probs_topics = sess.run([tf.argmax(model.summary_input_idxs, -1), model.probs_sent_topic_posterior], feed_dict=feed_dict)

    summary_sents = idxs_to_sents(summary_idxs[0], model.config)
    prob_topics = np.sum(probs_topics[0], 0) / np.sum(probs_topics[0]) # n_topic
    return summary_sents, prob_topics

def print_summary(instance, sess, model, summary_sents=None, prob_topics=None, parent_idx=0, depth=0, sort=True, console=True, beam=True, sample=True, logger=None):
    if logger is None:
        logger = logging.getLogger(model.config.fname_model)
        logger.setLevel(10)
        sh = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s:%(lineno)d:%(levelname)s:%(message)s')
        sh.setFormatter(formatter)
        logger.addHandler(sh)
        
    if summary_sents is None and prob_topics is None: # print root
        summary_sents, prob_topics = get_summary(instance, sess, model, beam=beam, sample=sample)

    if depth == 0: # print root
        if instance is not None:
            if 'summary' in instance.keys():
                logger.info('-----------Summary-----------')
                logger.info('SUMMARY: %s' % instance.summary)

        logger.info('-----------Topic sentences (beam: %s, sample: %s)-----------' % (beam, sample))
        if instance is not None: logger.info('Item idx: %s' % instance.business_id)
        pred_summary_sent = summary_sents[model.config.topic_idxs.index(parent_idx)]
        prob_topic = prob_topics[model.config.topic_idxs.index(parent_idx)]
        logger.info('%s P: %.3f SENT: %s' % (parent_idx, prob_topic, pred_summary_sent))

    child_idxs = model.config.tree_idxs[parent_idx]
    depth += 1
    for child_idx in child_idxs:
        pred_summary_sent = summary_sents[model.config.topic_idxs.index(child_idx)]
        prob_topic = prob_topics[model.config.topic_idxs.index(child_idx)]
        logger.info('  '*depth + '%i P: %.3f SENT: %s' % (child_idx, prob_topic, pred_summary_sent))
        if child_idx in model.config.tree_idxs: 
            print_summary(instance, sess, model, summary_sents, prob_topics, parent_idx=child_idx, depth=depth, console=False, logger=logger)
