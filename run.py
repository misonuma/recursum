import subprocess
import time
import _pickle as cPickle
from collections import defaultdict
import pdb
import logging
import multiprocessing
import numpy as np
import tensorflow as tf
import pandas as pd
from configure import update_config_tree, update_checkpoint
from tree import get_descendant_idxs
from evaluation.rouge_scorer import RougeScorer
# from evaluation.eval_utils import EvalMetrics
# from summarize import tmpsum
from summarize import treesum, greedysum
from IPython.display import clear_output
pd.set_option('display.max_columns', 50)

rouge_scorer = RougeScorer(rouge_types=['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)


def get_sents_from_tokens(tokens):
    return [' '.join(line_tokens) for line_tokens in tokens]

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

def init(config, trash=True):
    epoch = 0
    log_df = pd.DataFrame(columns=pd.MultiIndex.from_tuples(
        list(zip(*[['','','',
                    'TRAIN','','','','','','','','',
                    'VALID','','','','','','','','','','','','','','','','',''],
                    ['TIME','BETA','AGG',
                     'PPL','LVAE','LDISC','RECON','PRKL','SEKL','DISE','DITO','COVER',\
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

# +
def train(sess, model, saver, train_batches, dev_batches, test_batches, log_df, logger, sample=False, debug=False, jupyter=True):
    time_start = time.time()

    # train
    if not model.config.aggressive:
        ppl_losses_train, global_step, n_errors = compute_loss(sess, model, train_batches, mode='train', sample=True, debug=debug)
    else:
        ppl_losses_train, global_step, n_errors = compute_loss_aggressive(sess, model, train_batches, mode='train', sample=True, debug=debug)

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
    ppl_losses_dev, _, _ = compute_loss(sess, model, dev_batches, mode='eval', sample=sample)
    log_dev = ['%.2f' % np.minimum(loss, 1e+4) for loss in ppl_losses_dev]
    
#     ppl_losses_mi = compute_loss_mi(sess, model, dev_batches, mode='eval')
#     ppl_losses_mi = [0., 0., 0., 0.]
#     log_dev_mi = ['%.2f' % np.minimum(loss, 1e+4) for loss in ppl_losses_mi]
    
    depth_mean_logdetcovs_topic_posterior = compute_logdetcovs(sess, model, dev_batches, mode='eval', sample=sample)
    log_logdetcovs = ' | '.join(['%.1f' % logdetcov for logdetcov in depth_mean_logdetcovs_topic_posterior])
    ppl_dev = ppl_losses_dev[0]
    
#     topic_sents_list, probs_topic_list, _ = compute_topic_sents_probs(sess, model, dev_batches, mode='eval', sample=sample)
#     rouges_f1_dict = compute_rouges_f1(dev_batches, topic_sents_list, probs_topic_list, sys_sum=tmpsum, n_sents=5)
#     log_rouges = ['%.3f'%rouge for rouge in rouges_f1_dict.values()]

    summary_l_rouge_df = compute_rouges_f1(sess, model, dev_batches, sys_sum=greedysum, topk=4, threshold=0.6, truncate=4, max_summary_l=6, num_split=8)
    rouges = summary_l_rouge_df[max(summary_l_rouge_df.keys())].mean().to_dict().values()
    log_rouges = ['%.3f'%rouge for rouge in rouges]
    
    beta = sess.run(model.beta)
    log_beta = '%.3f'%beta
    log_agg = str(model.config.aggressive)
    
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
        log_df.loc[global_step] = pd.Series([log_time, log_beta, log_agg] + log_train + log_dev + log_rouges + [log_logdetcovs] + [log_errors], index=log_df.columns)
    except Exception as e:
        print(e)
        pdb.set_trace()
    if jupyter: display(log_df)
    log_df.to_pickle(model.config.path_log)
        
    # print sent
    test_instance = test_batches[0][0]
    logger.info('######################### Step: %i #########################'%global_step)
    print_summary(test_instance, sess, model, beam=True, sample=False, logger=logger)
    print_sample(test_instance, sess, model, logger=logger)
        
    return sess, model, saver, log_df, nan_flg


# -

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
                _, _, _, global_step, loss_list, loss_sum = \
                    sess.run([model.opt, model.opt_enc, model.opt_disc, tf.train.get_global_step(), model.loss_list_train, model.loss_sum], feed_dict = feed_dict, options=run_options)
            elif mode == 'eval':
                global_step = None
                loss_list, loss_sum = sess.run([model.loss_list_eval, model.loss_sum], feed_dict = feed_dict, options=run_options)
                    
        except tf.errors.InvalidArgumentError as ie:
            print(ie)
            continue

        except Exception as e:
            print(e)
            n_errors += 1
            bl, pr, co = check([tf.reduce_any(tf.is_nan(model.probs_topic_posterior)), \
                                    model.probs_topic_posterior, model.covs_topic_posterior], batch)                
        
        losses += [loss_list]
        loss_ppl += loss_sum # for computing PPL
        n_tokens += np.sum(feed_dict[model.t_variables['dec_sent_l']]) # for computing PPL

    losses_mean = list(np.mean(losses, 0))
    ppl = np.exp((loss_ppl)/n_tokens)
    ppl_losses = [ppl] + losses_mean
    
    return ppl_losses, global_step, n_errors

def compute_loss_aggressive(sess, model, batches, mode, sample):
    losses = []
    loss_ppl = 0
    n_tokens = 0
    n_errors = 0
    i_aggressive = 0
    
    for batch in batches:
        feed_dict = model.get_feed_dict(batch, mode=mode)
        feed_dict[model.t_variables['sample']] = sample
        run_options = tf.RunOptions(report_tensor_allocations_upon_oom=debug)
        
        try:
            if i_aggressive < model.config.n_aggressive:
                # only emb & disc
                _, _, global_step, loss_list, loss_sum = \
                    sess.run([model.opt_infer, model.opt_disc_infer, tf.train.get_global_step(), model.loss_list_train, model.loss_sum], feed_dict=feed_dict, options=run_options)
                i_aggressive += 1
            else:
                # only dec
                _, _, global_step, loss_list, loss_sum = \
                sess.run([model.opt_gen, model.opt_disc_gen, tf.train.get_global_step(), model.loss_list_train, model.loss_sum], feed_dict=feed_dict, options=run_options)
                i_aggressive = 0

        except Exception as e:
            print(e)
            n_errors += 1
            continue
        
        losses += [loss_list]
        loss_ppl += loss_sum # for computing PPL
        n_tokens += np.sum(feed_dict[model.t_variables['dec_sent_l']]) # for computing PPL

    losses_mean = list(np.mean(losses, 0))
    ppl = np.exp((loss_ppl)/n_tokens)
    ppl_losses = [ppl] + losses_mean
    
    return ppl_losses, global_step, n_errors


def compute_loss_mi(sess, model, batches, mode):
    def sample_gauss(means, logvars, partition_doc, n_sample=1):
        noises = np.random.standard_normal(size=(n_sample, means.shape[0], means.shape[1], means.shape[-1]))
        latents = means[None, :, :, :] + np.exp(0.5 * logvars)[None, :, :, :] * noises
        latents = latents[np.where(np.tile(partition_doc[None, :, :], (latents.shape[0], 1, 1)) > 0)]
        return latents

    pdf_diag = lambda latents, means, logvars: np.exp(-1/2*np.sum((latents[None, None, :, :]-means[:, :, None, :])**2/np.exp(logvars[:, :, None, :]), -1)) \
                                    / np.sqrt((2*np.pi)**means.shape[-1]*np.exp(np.sum(logvars, -1)))[:, :, None]
    pdf_full = lambda latents, means, covs: \
                                np.exp(-1/2*np.sum(np.matmul((latents[None, None, :, :]-means[:, :, None, :]), np.linalg.inv(covs))*(latents[None, None, :, :]-means[:, :, None, :]), -1))\
                                / np.sqrt((2*np.pi)**means.shape[-1]*np.linalg.det(covs))[:, :, None]

    means_sent_posterior, logvars_sent_posterior, probs_sent_topic_posterior, means_topic_posterior, covs_topic_posterior, loss_kl_sent_gmm, loss_kl_prob, partition_doc = \
        compute_tensors(sess, model, batches, \
            [model.means_sent_posterior, model.logvars_sent_posterior, model.probs_sent_topic_posterior, model.means_topic_posterior, model.covs_topic_posterior, model.loss_kl_sent_gmm, model.loss_kl_prob, tf.cast(model.mask_doc, dtype=tf.int32)], mode='eval')
    latents_sent_posterior = sample_gauss(means_sent_posterior, logvars_sent_posterior, partition_doc, model.config.n_sample)
    
    if model.config.latent:
        loss_kl_marg_prob = loss_mi_prob = 0.
    else:
        probs_sent_topic_prior = compute_tensors(sess, model, batches, [model.probs_sent_topic_prior], mode='eval')[0]
        # kl loss about the topic distribution
        probs_topic_posterior = np.mean(probs_sent_topic_posterior[np.where(partition_doc>0)], 0)
        logits_topic_posterior = np.log(probs_topic_posterior)
        logits_topic_prior = np.log(np.mean(probs_sent_topic_prior[np.where(partition_doc>0)], 0))

        loss_kl_marg_prob = probs_topic_posterior.dot(logits_topic_posterior - logits_topic_prior)
        loss_mi_prob = loss_kl_prob - loss_kl_marg_prob
    
    # kl loss about the latent code of topic sentences
    probs_sent_posterior = pdf_diag(latents_sent_posterior, means_sent_posterior, logvars_sent_posterior)
    logits_sent_posterior = np.log(np.mean(probs_sent_posterior[np.where(partition_doc>0)], 0))

    probs_sent_topic = pdf_full(latents_sent_posterior, means_topic_posterior, covs_topic_posterior)
    probs_sent_prior = np.matmul(probs_sent_topic_posterior, probs_sent_topic)
    logits_sent_prior = np.log(np.mean(probs_sent_prior[np.where(partition_doc>0)], 0))

    loss_kl_marg_sent_gmm = np.mean(logits_sent_posterior - logits_sent_prior)
    loss_mi_sent_gmm = loss_kl_sent_gmm - loss_kl_marg_sent_gmm
    
    loss_mi = loss_mi_prob + loss_mi_sent_gmm
    if loss_mi > model.config.max_loss_mi:
        model.config.max_loss_mi = loss_mi
        model.config.i_loss_mi = 0
    else:
        model.config.i_loss_mi += 1
        
    if model.config.i_loss_mi >= model.config.n_loss_mi:
        model.config.aggressive = False
    
    return loss_mi_prob, loss_kl_marg_prob, loss_mi_sent_gmm, loss_kl_marg_sent_gmm


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

def compute_tensors(sess, model, batches, tensors_list, mode, sample=None):
    batch_ndarrays_list = []
    for batch in batches:
        feed_dict = model.get_feed_dict(batch, mode=mode)
        if sample is not None: feed_dict[model.t_variables['sample']] = sample
        batch_ndarrays = sess.run(tensors_list, feed_dict=feed_dict)
        batch_ndarrays_list += [batch_ndarrays]
    
    ndarrays_list = []
    for i in range(len(tensors_list)):
        if len(batch_ndarrays_list[0][i].shape) > 1:
            ndarrays = tf.keras.preprocessing.sequence.pad_sequences([instance for batch_ndarrays in batch_ndarrays_list for instance in batch_ndarrays[i]], dtype=np.float32, padding='post')
        elif len(batch_ndarrays_list[0][i].shape) == 1:
            ndarrays = np.array([instance for batch_ndarrays in batch_ndarrays_list for instance in batch_ndarrays[i]])
        elif len(batch_ndarrays_list[0][i].shape) == 0:
            ndarrays = np.mean([batch_ndarrays[i] for batch_ndarrays, batch in zip(batch_ndarrays_list, batches) for _ in range(len(batch))])
        ndarrays_list += [ndarrays]
                                      
    return ndarrays_list


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

def compute_topic_posteriors_list(sess, model, batches, mode='eval', sample=False):
    means_topic_posterior_list, covs_topic_posterior_list, topic_sents_list = [], [], []
    for batch in batches:
        feed_dict = model.get_feed_dict(batch, mode=mode)
        feed_dict[model.t_variables['sample']] = sample
        
        means_topic_posterior, covs_topic_posterior, batch_topic_token_idxs = \
            sess.run([model.means_topic_posterior, model.covs_topic_posterior, model.beam_summary_idxs, ], \
                         feed_dict=feed_dict)
        means_topic_posterior_list += list(means_topic_posterior)
        covs_topic_posterior_list += list(covs_topic_posterior)
        topic_sents_list += [idxs_to_sents(topic_token_idxs, model.config) for topic_token_idxs in batch_topic_token_idxs]
    
    return means_topic_posterior_list, covs_topic_posterior_list, topic_sents_list


def compute_rouges_f1(sess, model, batches, sys_sum, topk, threshold, truncate, max_summary_l, num_split):
    data_df = pd.DataFrame([instance for batch in batches for instance in batch])
    topic_sents_list, _, _ = compute_topic_sents_probs(sess, model, batches, mode='eval', sample=False)
    text_list = [row.text.replace('\n', '') for _, row in data_df.iterrows()]
    verbose = False
    
    args = [(model.config.tree_idxs, topic_sents, text, topk, threshold, truncate, max_summary_l, verbose) for topic_sents, text in zip(topic_sents_list, text_list)]
    if num_split > 1:
        pool = multiprocessing.Pool(processes=num_split)
        summary_l_sents_list = pool.map(sys_sum, args)
        pool.close()
    else:
        summary_l_sents_list = []
        for arg in args:
            summary_l_sents_list += sys_sum(arg)
    
    summary_l_rouge_df = {}
    for summary_l in range(1, max_summary_l+1):
        summary_list = [get_text_from_sents(summary_l_sents[summary_l]['sents']) for summary_l_sents in summary_l_sents_list]
        reference_list = [row.summary for _, row in data_df.iterrows()]
        summary_l_rouge_df[summary_l] = compute_rouge(data_df, reference_list, summary_list)
    return summary_l_rouge_df


def compute_rouge(data_df, reference_list, summary_list):
    assert len(data_df) == len(reference_list) == len(summary_list)
    rouge_dict = {index: {rouge_name: getattr(rouge_obj, 'fmeasure') \
                              for rouge_name, rouge_obj in rouge_scorer.score(target=reference, prediction=summary).items()}\
                              for index, reference, summary in zip(data_df.index, reference_list, summary_list)}
    rouge_df = pd.DataFrame.from_dict(rouge_dict, orient='index')
    return rouge_df


# +
# def compute_rouges_f1(batches, topic_sents_list, probs_topic_list, sys_sum, n_sents):
#     sys_summary_sents_list = [sys_sum(topic_sents, probs_topic, n_sents=n_sents) for topic_sents, probs_topic in zip(topic_sents_list, probs_topic_list)]
#     sys_summaries = ['. '.join(sys_summary_sents) for sys_summary_sents in sys_summary_sents_list]
#     ref_summaries = [instance.summary for batch in batches for instance in batch]
#     eval_metrics = EvalMetrics()
#     rouges_f1_dict = eval_metrics.calc_rouges_f1_dict(sys_summaries, ref_summaries)
#     return rouges_f1_dict
# -

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
