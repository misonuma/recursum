#coding: utf-8
import os
import json
import pdb
import multiprocessing
import argparse
from collections import defaultdict
import _pickle as cPickle

import tensorflow as tf
import numpy as np
import pandas as pd
from nltk import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity

from configure import get_config, update_config
from data_structure import get_batches, get_batches_iterator
from run import idxs_to_sents, get_text_from_sents, compute_rouges_f1, compute_rouge, compute_topic_sents_probs, get_summary
from summarize import greedysum


def refine_text(text):
    refine_sents = []
    sents = text.split('.\n')
    for sent in sents:
        if sent == '': continue
        refine_sent = ''
        words = sent.split()
        for index, word in enumerate(words):            
            if index==0 or word == 'i': word = word.capitalize()
            
            if index==0 or "'" in word or word == ',':
                refine_sent += word
            else:
                refine_sent += ' ' + word
        
        refine_sent += '.'
        refine_sents.append(refine_sent)
        
    refine_text = ' '.join(refine_sents)
    return refine_text

def get_evaluation_config(data):
    parser = argparse.ArgumentParser()

    parser.add_argument('-seed', type=int, default=1234)
    parser.add_argument('-batch_size', type=int, default=4)

    config = parser.parse_args('')
    config.data = data
    config.path_data = os.path.join('data', config.data, '%s_df.pkl' % config.data)
    config.dir_eval = os.path.join('eval', config.data)
    config.dir_ref = os.path.join(config.dir_eval, 'ref')
    config.path_meansum_raw = os.path.join(config.dir_eval, 'meansum_df.pkl')
    config.path_copycat_raw = os.path.join(config.dir_eval, 'copycat.json')
    config.path_denoisesum_raw = os.path.join(config.dir_eval, 'denoisesum.json')
    
    config.path_lead = os.path.join(config.dir_eval, 'lead_df.pkl')
    config.path_opinosis = os.path.join(config.dir_eval, 'opinosis_df.pkl')
    config.path_lexrank = os.path.join(config.dir_eval, 'lexrank_df.pkl')
    config.path_meansum = os.path.join(config.dir_eval, 'meansum_df.pkl')
    config.path_copycat = os.path.join(config.dir_eval, 'copycat_df.pkl')
    config.path_recursum = os.path.join(config.dir_eval, 'recursum_df.pkl')
    
    config.path_recursum_nondisc = os.path.join(config.dir_eval, 'recursum_nondisc_df.pkl')
    config.path_recursum_nonatt = os.path.join(config.dir_eval, 'recursum_nonatt_df.pkl')
    config.path_recursum_nonnucleus = os.path.join(config.dir_eval, 'recursum_nonnucleus_df.pkl')
    
    config.path_recursum_22 = os.path.join(config.dir_eval, 'recursum_22_df.pkl')
    config.path_recursum_33 = os.path.join(config.dir_eval, 'recursum_33_df.pkl')
    config.path_recursum_44 = os.path.join(config.dir_eval, 'recursum_44_df.pkl')
    config.path_recursum_55 = os.path.join(config.dir_eval, 'recursum_55_df.pkl')
    config.path_recursum_66 = os.path.join(config.dir_eval, 'recursum_66_df.pkl')
    
    config.path_recursum_3 = os.path.join(config.dir_eval, 'recursum_3_df.pkl')
    config.path_recursum_333 = os.path.join(config.dir_eval, 'recursum_333_df.pkl')
    
    config.path_oracle = os.path.join(config.dir_eval, 'oracle_df.pkl')
    
    config.path_eval_quality_df = os.path.join(config.dir_eval, 'quality', 'eval_quality_%s_df'%config.data)
    config.path_eval_faithfulness_df = os.path.join(config.dir_eval, 'faithfulness', 'eval_faithfulness_%s_df'%config.data)
    config.path_eval_coverage_df = os.path.join(config.dir_eval, 'coverage', 'eval_coverage_%s_df'%config.data)
    
    config.path_affinity = os.path.join(config.dir_eval, 'affinity_df.pkl')
    config.path_affinity_tmp = os.path.join(config.dir_eval, 'affinity_tmp_df.pkl')
    config.path_affinity_reference = os.path.join(config.dir_eval, 'affinity_reference_df.pkl')
    config.path_affinity_text = os.path.join(config.dir_eval, 'affinity_text_df.pkl')
    config.path_eval_affinity = os.path.join(config.dir_eval, 'eval_affinity_df.pkl')
    config.path_specificity = os.path.join(config.dir_eval, 'specificity_df.pkl')    
    config.path_specificity_tmp = os.path.join(config.dir_eval, 'specificity_tmp_df.pkl')
    config.path_specificity_reference = os.path.join(config.dir_eval, 'specificity_reference_df.pkl')
    config.path_specificity_text = os.path.join(config.dir_eval, 'specificity_text_df.pkl')
    config.path_eval_specificity = os.path.join(config.dir_eval, 'eval_specificity_df.pkl')
    config.path_latenttree = os.path.join(config.dir_eval, 'latenttree_df.pkl')
    config.path_latenttree_tmp = os.path.join(config.dir_eval, 'latenttree_tmp_df.pkl')
    config.path_latenttree_tmp0 = os.path.join(config.dir_eval, 'latenttree_tmp0_df.pkl')
    config.path_latenttree_tmp1 = os.path.join(config.dir_eval, 'latenttree_tmp1_df.pkl')
    config.path_eval_latenttree = os.path.join(config.dir_eval, 'eval_latenttree_df.pkl')
    
    config.path_config = os.path.join(config.dir_eval, 'config.pkl')
    config.path_stopwords = os.path.join('data', 'stopwords_mallet.txt')

    config.PAD = '<pad>' # This has a vocab id, which is used to pad the encoder input, decoder input and target sequence
    config.UNK = '<unk>' # This has a vocab id, which is used to represent out-of-vocabulary words
    config.BOS = '<p>' # This has a vocab id, which is used at the beginning of every decoder input sequence
    config.EOS = '</p>' # This has a vocab id, which is used at the end of untruncated target sequences
    return config

def load_model(nb_name, train_batches, dev_batches, word_to_idx, bow_idxs, nucleus=None, seed=None, backup=False):
    config = get_config(nb_name)
    config = update_config(config, train_batches, dev_batches, word_to_idx, bow_idxs)
    if nucleus is not None: config.nucleus = nucleus
    if seed is not None: config.seed = seed
    if backup: config.dir_model = config.dir_model.replace('atttglm/', 'backup/')

    model = config.Model(config)
    return model

def restore_model(model, model_path):
    sess = tf.Session()
    saver = tf.train.Saver(max_to_keep=model.config.max_to_keep)
    saver.restore(sess, model_path)
    return sess

def evaluate_rouges(test_batches, train_batches, dev_batches, word_to_idx, bow_idxs, sys_sum, nb_name_list, n_path_list, topk_list, threshold_list, truncate_list, max_summary_l, num_split=25, new=False):
    rouges_list = []
    assert len(nb_name_list) == len(n_path_list)
    for nb_name, n_path in zip(nb_name_list, n_path_list):
        print('loading... %s' % nb_name)
        model = load_model(nb_name, train_batches, dev_batches, word_to_idx, bow_idxs, new=new)
        ckpt = tf.train.get_checkpoint_state(model.config.dir_model)
        all_model_paths = ckpt.all_model_checkpoint_paths
        try:
            model_path = all_model_paths[n_path]
            sess = restore_model(model, model_path)
            for topk in topk_list:
                for threshold in threshold_list:
                    for truncate in truncate_list:
                        summary_l_rouge_df = compute_rouges_f1(sess, model, test_batches, sys_sum=sys_sum, topk=topk, threshold=threshold, \
                                                                                                        truncate=truncate, max_summary_l=max_summary_l, num_split=num_split)
                        for summary_l, rouge_df in summary_l_rouge_df.items():
                            mean_rouge_df = np.mean(rouge_df)
                            rouges_list.append({'nb_name': nb_name,
                                                       'n_path': n_path,
                                                       'top_k': topk,
                                                       'threshold': threshold,
                                                       'truncate': truncate,
                                                       'summary_l': summary_l,
                                                       'rouge1': mean_rouge_df['rouge1'],
                                                       'rouge2': mean_rouge_df['rouge2'],
                                                       'rougeL': mean_rouge_df['rougeL']
                                                      })
            sess.close()
        except Exception as e:
            print(e)
            pdb.set_trace()
            continue
    rouges_df = pd.DataFrame(rouges_list)
    return rouges_df

def get_recursum_df(sess, model, test_df, sys_sum, topk, threshold, truncate, summary_l, num_split):    
    data_df = test_df.groupby('business_id').agg({
        'doc_l': lambda doc_l_series: doc_l_series.values[0],
        'sent_l': lambda sent_l_series: sent_l_series.values[0],
        'token_idxs': lambda token_idxs_series: token_idxs_series.values[0],
        'text': lambda text_series: text_series.values[0]
    })

    batches = get_batches(data_df, model.config.batch_size)
    topic_sents_list, probs_topic_list, topic_tokens_list = compute_topic_sents_probs(sess, model, batches, mode='eval', sample=False)
    text_list = [row.text.replace('\n', '') for _, row in data_df.iterrows()]
    verbose = False

    args = [(model.config.tree_idxs, topic_sents, text, topk, threshold, truncate, summary_l, verbose) for topic_sents, text in zip(topic_sents_list, text_list)]
    pool = multiprocessing.Pool(processes=num_split)
    summary_l_sents_list = pool.map(sys_sum, args)
    pool.close()

    summary_list = [get_text_from_sents(summary_l_sents[summary_l]['sents']) for summary_l_sents in summary_l_sents_list]
    summary_idxs_list = [[model.config.topic_idxs[topic_index] for topic_index in summary_l_sents[summary_l]['indices']] \
                                             for summary_l_sents in summary_l_sents_list]

    data_df['recursum'] = summary_list
    data_df['summary_idxs'] = summary_idxs_list
    data_df['topic_sents'] = topic_sents_list
    data_df['topic_tokens'] = topic_tokens_list
    data_df['probs_topic'] = probs_topic_list

    recursum_df = pd.merge(test_df, data_df[['recursum', 'summary_idxs', 'topic_sents', 'topic_tokens', 'probs_topic']], on='business_id', how='left')
    recursum_df[['rouge1', 'rouge2', 'rougeL']] = compute_rouge(recursum_df, \
                                                    reference_list=list(recursum_df.summary), summary_list=list(recursum_df.recursum))
    recursum_df = recursum_df.set_index(test_df.index)
    assert recursum_df['business_id'].to_dict() == test_df['business_id'].to_dict()
    return recursum_df

def get_denoisesum_df(config):
    denoisesum_df = pd.read_json(config.path_denoisesum_raw)
    denoisesum_df[['rouge1', 'rouge2', 'rougeL']] = compute_rouge(denoisesum_df, reference_list=list(denoisesum_df.summary), summary_list=list(denoisesum_df.denoisesum))
    return denoisesum_df


def get_meansum_df(config):
    meansum_df = pd.read_pickle(config.path_meansum_raw)
    meansum_df[['rouge1', 'rouge2', 'rougeL']] = compute_rouge(meansum_df, reference_list=list(meansum_df.summary), summary_list=list(meansum_df.meansum))
    meansum_df.to_pickle(config.path_meansum)
    return meansum_df

def get_copycat_df(data_df, config):
    business_id_dict = {
        '#NAME1': '-zbcosKSMGDhaZYN-CrcVA',
        '#NAME2': '-i3pCgQi_Y9NiSSWs6G7bw',
        '#NAME3': '-_TSaVr53qiEGqMkwyEMaQ',
        '#NAME4': '-vCLrTTgw6pBufdarW8ynA',
        '#NAME5': '-K3kqmykKlhlB4arCsLHOw',
        '#NAME6': '-exEWEQ3iSMVC-QUP_ycPQ',
        '#NAME7': '-_yEVC3_3M6YOsamYfNFEw',
        '#NAME8': '-NR4KqS6lHseNvJ-GFzfMA',
        '#NAME9': '-ot4Xd6GxSUOqwUj7okZuA',
        '#NAME10': '-pV9kWNoA9vyHfM_auYecA',
        '#NAME11': '-FNquqGseSCVMWo7KbK-Tg',
        '#NAME12': '-Qkx7W0itbAApcG5lJuMFQ',
        '#NAME13': '-SJcjOv88ZHjIU44U4vWTQ',
        '#NAME14': '-isxnIljKLVjc9qEhCiaGg',
        '#NAME15': '-iPc_YSSqvM1CpZxxeUTXw',
        '#NAME16': '-ADtl9bLp8wNqYX1k3KuxA',
        '#NAME17': '-zEpEmDfFQL-ph0N3BDlXA',
        '#NAME18': '-oOKqZbYDt08zaWWyLZNIw',
        '#NAME19': '-PbCfkydmvuNcG9VG_ixkQ',
        '#NAME20': '-pN44P-_PjRpcj4Rk2wMOg',
        '#NAME21': '-dcI8oWvxdMCGp00da8Ksg',
        '#NAME22': '-MKWJZnMjSit406AUKf7Pg'
    }
    
    json_copycat = json.load(open(config.path_copycat_raw))
    json_copycat_df = pd.concat([pd.DataFrame.from_dict(json_copycat[category], orient='index') for category in list(json_copycat)[:-1]])
    if config.data == 'yelp': json_copycat_df.index = [business_id_dict[raw_id] if 'NAME' in raw_id else raw_id for raw_id in json_copycat_df.index]
    
    copycat_df = pd.merge(data_df, \
                                             pd.DataFrame([{'business_id': index, 'copycat': ' '.join(copycat_summary)} \
                                                                           for index, copycat_summary_list in json_copycat_df.gen_summ.to_dict().items() \
                                                                           for copycat_summary in copycat_summary_list]), how='left')
    copycat_df[['rouge1', 'rouge2', 'rougeL']] = compute_rouge(copycat_df, reference_list=list(copycat_df.summary), summary_list=list(copycat_df.copycat))
    copycat_df.index = data_df.index
    assert len(data_df) == len(copycat_df)
    copycat_df.to_pickle(config.path_copycat)
    return copycat_df

def get_lead_df(data_df, config):
    lead_df = data_df.copy()
    lead_df['lead'] = lead_df.text.apply(lambda text: ' '.join([sent_tokenize(doc)[0].strip() for doc in text.split('</DOC>')]))
    lead_df[['rouge1', 'rouge2', 'rougeL']] = \
        compute_rouge(lead_df, reference_list=list(lead_df.summary), summary_list=list(lead_df.lead))
    lead_df.to_pickle(config.path_lead)
    return lead_df

def get_lexrank_df(data_df, n_sents, min_sent_l, config):
    def lexrank(tfidfbows):
        cos_matrix = cosine_similarity(tfidfbows.toarray(), tfidfbows.toarray())
        eig_values, _ = np.linalg.eig(cos_matrix)
        eig_indices = np.argsort(eig_values)[::-1]
        return eig_indices
    
    lexrank_df = data_df.copy()
    lexrank_df['eig_indices'] = lexrank_df['tfidfbows'].apply(lexrank)
    lexrank_df['length_indices'] = data_df.sent_l.apply(lambda sent_l: np.where(np.array(sent_l) > min_sent_l)[0])
    lexrank_df['summary_indices'] = lexrank_df.apply(lambda row: [i for i in row.eig_indices if i in row.length_indices][:n_sents], 1)
    
    lexrank_df['lexrank'] = lexrank_df.apply(lambda row: \
                                                           get_text_from_sents(idxs_to_sents(np.array(row.token_idxs)[row.summary_indices], config)), 1)
    lexrank_df[['rouge1', 'rouge2', 'rougeL']] = compute_rouge(lexrank_df, \
                                                reference_list=list(lexrank_df.summary), summary_list=list(lexrank_df.lexrank))
    lexrank_df.to_pickle(config.path_lexrank)
    return lexrank_df

def get_opinosis_df(test_df, dev_df, config):
    opinosis_df = test_df.copy()
    
    max_summary_list = [4, 5, 6]
    redundancy_list = [0, 1]
    gap_list = [2, 3, 4]
    run_id_rougeL = {}
    for max_summary in max_summary_list:
        for redundancy in redundancy_list:
            for gap in gap_list:
                run_id='max-%i-red-%i-gap-%i' % (max_summary, redundancy, gap)
                dir_dev = os.path.join('eval', config.data, 'opinosis', 'dev', 'output', run_id)
                paths_dev = [os.path.join(dir_dev, '{0:03}.{1}.system'.format(i, run_id)) for i in dev_df.index]
                summary_list = []
                for path_dev in paths_dev:
                    with open(path_dev, 'r') as f:
                        summary = refine_text(f.read())
                    summary_list += [summary]

                rougeL = compute_rouge(dev_df, list(dev_df.summary), summary_list).mean()['rougeL']
                run_id_rougeL[run_id] = rougeL
                
    run_id_test = sorted(run_id_rougeL.items(), key=lambda x:x[1], reverse=True)[0][0]
    dir_test = os.path.join('eval', config.data, 'opinosis', 'test', 'output', run_id_test)
    paths_test = [os.path.join(dir_test, '{0:03}.{1}.system'.format(i, run_id_test)) for i in opinosis_df.index]
    summary_list = []
    for path_test in paths_test:
        with open(path_test, 'r') as f:
            summary = refine_text(f.read())
        summary_list += [summary]

    opinosis_df[['rouge1', 'rouge2', 'rougeL']] = compute_rouge(opinosis_df, list(opinosis_df.summary), summary_list)
    opinosis_df.to_pickle(config.path_opinosis)
    return opinosis_df

def approx_rand_test(values0, values1, n_perm=10000):
    values = np.concatenate([values0[None, :], values1[None, :]], 0)
    n_val = values.shape[1]
    
    assert len(values0) == len(values1)
    diff_mean = np.abs(np.mean(values0-values1))

    perms = np.random.randint(2, size=(n_perm, n_val))
    reverse_perms = 1 - perms

    perm_values = values[np.concatenate([perms[None, :], reverse_perms[None, :]], 0), np.tile(np.arange(n_val), [2, n_perm, 1])]
    assert np.all(np.sum(perm_values, 0)[0] == np.sum(values, 0))

    diffs_perm = np.abs(np.mean(perm_values[0], -1) - np.mean(perm_values[1], -1))
    pvalue = np.sum(diffs_perm >= diff_mean) / (n_perm+1)
    return pvalue

def print_recursum(instance, sess=None, model=None, model_config=None, summary_idxs=None, topic_sents=None, prob_topics=None, parent_idx=0, depth=0):
    if topic_sents is None and prob_topics is None: # print root
        topic_sents, prob_topics = get_summary(instance, sess, model, beam=True, sample=False)
        
    if model_config is None: model_config = model.config

    if summary_idxs is None:
        topk=4
        threshold=0.6
        truncate=0
        max_summary_l=6
        verbose=False
        summary_l_sents = greedysum([model_config.tree_idxs, topic_sents, instance.text, topk, threshold, truncate, max_summary_l, verbose])
        summary = ' '.join(summary_l_sents[max(summary_l_sents)]['sents'])
        print(summary)
        summary_indices = summary_l_sents[max(summary_l_sents)]['indices']
        summary_idxs = [model_config.topic_idxs[index] for index in summary_indices]
        
    if depth == 0: # print root
        print('-----------RecurSum-----------')
        pred_summary_sent = topic_sents[model_config.topic_idxs.index(parent_idx)]
        prob_topic = prob_topics[model_config.topic_idxs.index(parent_idx)]
        tag = '<summary> ' if parent_idx in summary_idxs else ''
        print('%s P: %.3f, SENT: %s' % (parent_idx, prob_topic, tag+pred_summary_sent))

    child_idxs = model_config.tree_idxs[parent_idx]
    depth += 1
    for child_idx in child_idxs:
        pred_summary_sent = topic_sents[model_config.topic_idxs.index(child_idx)]
        prob_topic = prob_topics[model_config.topic_idxs.index(child_idx)]
        tag = '<summary> ' if child_idx in summary_idxs else ''
        print('  '*depth + '%i P: %.3f, SENT: %s' % (child_idx, prob_topic, tag+pred_summary_sent))
            
        if child_idx in model_config.tree_idxs: 
            print_recursum(instance, sess, model, model_config, summary_idxs=summary_idxs, topic_sents=topic_sents, prob_topics=prob_topics, parent_idx=child_idx, depth=depth)
