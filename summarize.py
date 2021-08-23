import numpy as np

from evaluation.rouge_scorer import RougeScorer
from tree import get_topic_idxs


def get_sorted_topic_idxs(tree_idxs, parent_idx=0, sorted_topic_idxs=None):
    if sorted_topic_idxs is None: sorted_topic_idxs = []
    child_idxs = tree_idxs[parent_idx]
    sorted_topic_idxs.append(parent_idx)
    for child_idx in child_idxs:
        
        if child_idx in tree_idxs:
            sorted_topic_idxs = get_sorted_topic_idxs(tree_idxs, parent_idx=child_idx, sorted_topic_idxs=sorted_topic_idxs)
        else:
            sorted_topic_idxs.append(child_idx)

    return sorted_topic_idxs

def greedysum(args):
    tree_idxs, topic_sents, text, topk, threshold, max_summary_l = args
    
    docs = [doc.strip() for doc in text.split('</DOC>')]
    assert len(docs) == 8

    topic_idxs = get_topic_idxs(tree_idxs)
    summary_l_sents = {}

    topk_summary_sents_list = [[]]
    topk_summary_indices_list = [[]]
    topk_summary_rouge_list = [0.]

    unique_topic_sents, unique_topic_indices = np.unique(topic_sents, return_index=True)
    candidate_topic_indices_list = [[i for i, topic_sent in enumerate(unique_topic_sents)]]
    summary_sents = []
    
    rouge_name = 'rouge1'
    stopwords = None
    rouge_scorer = RougeScorer(rouge_types=list(set([rouge_name])), use_stemmer=True, stopwords=stopwords)

    def mean_rouge(docs, candidate_summary_sents):
        return np.mean([getattr(rouge_scorer.score(target=doc, prediction=' '.join(candidate_summary_sents))[rouge_name], 'fmeasure') for doc in docs])

    def rouge_precision(topk_summary_sents, topic_sent):
        return getattr(rouge_scorer.score(target=' '.join(topk_summary_sents), prediction=topic_sent)[rouge_name], 'precision')
    
    sorted_summary_sents = ['']
    sorted_summary_indices = []
    for summary_l in range(1, max_summary_l+1):
        if sum([len(candidate_topic_indices) for candidate_topic_indices in candidate_topic_indices_list]) == 0: 
            summary_l_sents[summary_l] = {'sents': sorted_summary_sents, 'indices': sorted_summary_indices}
            continue
            
        # compute rouge for each candidate summary
        candidate_topic_sents_list = [[unique_topic_sents[topic_index] for topic_index in candidate_topic_indices] \
                                                                              for candidate_topic_indices in candidate_topic_indices_list]
        candidate_summaries_rouge_list = [[mean_rouge(docs, candidate_summary_sents=topk_summary_sents + [topic_sent]) \
                                                       for topic_sent in candidate_topic_sents] \
                                                       for topk_summary_sents, candidate_topic_sents in zip(topk_summary_sents_list, candidate_topic_sents_list)]
        assert len(topk_summary_sents_list) == len(candidate_topic_sents_list) == len(topk_summary_indices_list) == len(candidate_topic_indices_list) == len(topk_summary_rouge_list) == len(candidate_summaries_rouge_list)

        candidate_summaries_sents_indices_list = [[
            {'sents': topk_summary_sents + [topic_sent], 'indices': topk_summary_indices + [topic_index], 'topk_topic_index': topic_index, 'rouge': candidate_summary_rouge}
            if rouge_precision(topk_summary_sents, topic_sent) <= threshold
            else {'sents': topk_summary_sents, 'indices': topk_summary_indices, 'topk_topic_index': None, 'rouge': topk_summary_rouge}
            for topic_sent, topic_index, candidate_summary_rouge in zip(candidate_topic_sents, candidate_topic_indices, candidate_summaries_rouge)] \
        for topk_summary_sents, candidate_topic_sents, topk_summary_indices, candidate_topic_indices, topk_summary_rouge, candidate_summaries_rouge \
        in zip(topk_summary_sents_list, candidate_topic_sents_list, topk_summary_indices_list, candidate_topic_indices_list, topk_summary_rouge_list, candidate_summaries_rouge_list)]

        candidate_summaries_sents_list = [[candidate_summary_sents_indices['sents']\
                                                      for candidate_summary_sents_indices in candidate_summaries_sents_indices]\
                                                      for candidate_summaries_sents_indices in candidate_summaries_sents_indices_list]
        candidate_summaries_indices_list = [[candidate_summary_sents_indices['indices']\
                                                      for candidate_summary_sents_indices in candidate_summaries_sents_indices]\
                                                      for candidate_summaries_sents_indices in candidate_summaries_sents_indices_list]
        candidate_summaries_rouge_list = [[candidate_summary_sents_indices['rouge']\
                                                      for candidate_summary_sents_indices in candidate_summaries_sents_indices]\
                                                      for candidate_summaries_sents_indices in candidate_summaries_sents_indices_list]
        topk_topic_indices_list = [[candidate_summary_sents_indices['topk_topic_index']\
                                                      for candidate_summary_sents_indices in candidate_summaries_sents_indices]\
                                                      for candidate_summaries_sents_indices in candidate_summaries_sents_indices_list]

        candidate_rouges = [candidate_summary_rouge
                                                      for candidate_summaries_rouge in candidate_summaries_rouge_list\
                                                      for candidate_summary_rouge in candidate_summaries_rouge]

        # identify top k rouge of candidate summaries
        candidate_topic_args = np.array([[i, j] for i in range(len(candidate_summaries_sents_list)) \
                                                         for j in range(len(candidate_summaries_sents_list[i]))])
        assert len(candidate_topic_args) == len(candidate_rouges)
        topk_topic_args = candidate_topic_args[np.argsort(candidate_rouges)[::-1]]

        # identify top k rouge of candidate summaries
        topk_summary_sents_list = []
        topk_summary_indices_list = []
        topk_summary_rouge_list = []
        new_candidate_topic_indices_list = []
        for topk_topic_arg in topk_topic_args:
            topk_summary_indices = candidate_summaries_indices_list[topk_topic_arg[0]][topk_topic_arg[1]]
            if set(topk_summary_indices) in [set(indices) for indices in topk_summary_indices_list]: continue

            topk_summary_indices_list += [topk_summary_indices]
            topk_summary_sents_list += [candidate_summaries_sents_list[topk_topic_arg[0]][topk_topic_arg[1]]]
            topk_summary_rouge_list += [candidate_summaries_rouge_list[topk_topic_arg[0]][topk_topic_arg[1]]]
            topk_topic_index = topk_topic_indices_list[topk_topic_arg[0]][topk_topic_arg[1]]

            candidate_topic_indices = list(candidate_topic_indices_list[topk_topic_arg[0]])
            if topk_topic_index is not None:
                candidate_topic_indices.remove(topk_topic_index)
                new_candidate_topic_indices_list += [candidate_topic_indices]
            else:
                new_candidate_topic_indices_list += [candidate_topic_indices]

            if len(topk_summary_indices_list) >= topk: break

        candidate_topic_indices_list = new_candidate_topic_indices_list
        summary_sents = topk_summary_sents_list[0]
        summary_indices = list(unique_topic_indices[topk_summary_indices_list[0]])

        sorted_topic_idxs = get_sorted_topic_idxs(tree_idxs)
        sorted_topic_indices = [topic_idxs.index(topic_idx) for topic_idx in sorted_topic_idxs]

        sorted_summary_indices = []
        sorted_summary_sents = []
        for sorted_topic_index in sorted_topic_indices:
            if sorted_topic_index in summary_indices:
                sorted_summary_indices.append(sorted_topic_index)
                sorted_summary_sent = summary_sents[summary_indices.index(sorted_topic_index)]
                sorted_summary_sents.append(sorted_summary_sent)

        summary_l_sents[summary_l] = {'sents': sorted_summary_sents, 'indices': sorted_summary_indices}
        
    return summary_l_sents
