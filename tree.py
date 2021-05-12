#coding: utf-8
import numpy as np
import tensorflow as tf

def get_topic_idxs(tree_idxs):
    return [0] + [idx for child_idxs in tree_idxs.values() for idx in child_idxs]

def get_child_to_parent_idxs(tree_idxs):
    return {child_idx: parent_idx for parent_idx, child_idxs in tree_idxs.items() for child_idx in child_idxs}

def get_depth(tree_idxs, parent_idx=0, tree_depth=None, depth=1):
    if tree_depth is None: tree_depth={0: depth}

    child_idxs = tree_idxs[parent_idx]
    depth +=1
    for child_idx in child_idxs:
        tree_depth[child_idx] = depth
        if child_idx in tree_idxs: get_depth(tree_idxs, child_idx, tree_depth, depth)
    return tree_depth

def get_ancestor_idxs(leaf_idx, child_to_parent_idxs, ancestor_idxs = None):
    if ancestor_idxs is None: ancestor_idxs = [leaf_idx]
    parent_idx = child_to_parent_idxs[leaf_idx]
    ancestor_idxs += [parent_idx]
    if parent_idx in child_to_parent_idxs: get_ancestor_idxs(parent_idx, child_to_parent_idxs, ancestor_idxs)
    return ancestor_idxs[::-1]

def get_descendant_idxs(tree_idxs, parent_idx, descendant_idxs = None):
    if descendant_idxs is None: descendant_idxs = [parent_idx]

    if parent_idx in tree_idxs:
        child_idxs = tree_idxs[parent_idx]
        descendant_idxs += child_idxs
        for child_idx in child_idxs:
            if child_idx in tree_idxs: get_descendant_idxs(tree_idxs, child_idx, descendant_idxs)
    return descendant_idxs

def get_mask_tree_reg(tree_idxs, all_child_idxs):
    mask_tree = np.zeros([len(all_child_idxs), len(all_child_idxs)], dtype=np.float32)
    for parent_idx, child_idxs in tree_idxs.items():
        neighbor_idxs = child_idxs
        for neighbor_idx1 in neighbor_idxs:
            for neighbor_idx2 in neighbor_idxs:
                neighbor_index1 = all_child_idxs.index(neighbor_idx1)
                neighbor_index2 = all_child_idxs.index(neighbor_idx2)
                mask_tree[neighbor_index1, neighbor_index2] = mask_tree[neighbor_index2, neighbor_index1] = 1.
    return mask_tree

def get_mask_tree(tree_idxs, topic_idxs, depth_topic_idxs):
    mask_tree = np.zeros([len(topic_idxs), len(topic_idxs)], dtype=np.float32)
    for depth in range(1, max(depth_topic_idxs.keys())):
        parent_idxs = depth_topic_idxs[depth]
        all_child_idxs = [child_idx for parent_idx in parent_idxs for child_idx in tree_idxs[parent_idx]]
        for parent_idx in parent_idxs:
            child_idxs = tree_idxs[parent_idx]
            other_child_idxs = set(all_child_idxs) - set(child_idxs)
            for other_child_idx in other_child_idxs:
                mask_tree[topic_idxs.index(parent_idx), topic_idxs.index(other_child_idx)] = 1.
    return mask_tree

def get_mask_tree_sibling(tree_idxs, topic_idxs, depth_topic_idxs):
    mask_tree = get_mask_tree(tree_idxs, topic_idxs, depth_topic_idxs)
    for parent_idx, child_idxs in tree_idxs.items():
        for child_idx1 in child_idxs:
            for child_idx2 in child_idxs:
                if child_idx1 != child_idx2:
                    mask_tree[topic_idxs.index(child_idx1), topic_idxs.index(child_idx2)] = 1.
    return mask_tree

def get_mask_tree_other(tree_idxs, topic_idxs):
    mask_tree = np.ones([len(topic_idxs), len(topic_idxs)], dtype=np.float32)
    parent_to_descendant_idxs = {parent_idx: get_descendant_idxs(tree_idxs, parent_idx) for parent_idx in tree_idxs}
    
    for parent_idx, descendant_idxs in parent_to_descendant_idxs.items():
        for descendant_idx in descendant_idxs:
            mask_tree[topic_idxs.index(parent_idx), topic_idxs.index(descendant_idx)] = mask_tree[topic_idxs.index(descendant_idx), topic_idxs.index(parent_idx)] = 0.
            
    return mask_tree

def get_mask_tree_reverse(tree_idxs, topic_idxs, depth_topic_idxs):
    mask_tree = np.zeros([len(topic_idxs), len(topic_idxs)], dtype=np.float32)
    for depth in range(1, max(depth_topic_idxs.keys())):
        parent_idxs = depth_topic_idxs[depth]
        all_child_idxs = [child_idx for parent_idx in parent_idxs for child_idx in tree_idxs[parent_idx]]
        for parent_idx in parent_idxs:
            child_idxs = tree_idxs[parent_idx]
            other_child_idxs = set(all_child_idxs) - set(child_idxs)
            for other_child_idx in other_child_idxs:
                mask_tree[topic_idxs.index(other_child_idx), topic_idxs.index(parent_idx)] = 1.
    return mask_tree
