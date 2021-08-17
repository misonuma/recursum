#coding: utf-8
from collections import defaultdict
import numpy as np
import tensorflow as tf
import pdb

from tree import get_ancestor_idxs

class DoublyGRUCell:
    def __init__(self, dim_hidden, output_layer=None):
        self.dim_hidden = dim_hidden
        
        self.ancestral_layer = tf.layers.Dense(units=dim_hidden, activation=tf.nn.tanh, name='ancestral')
        self.fraternal_layer = tf.layers.Dense(units=dim_hidden, activation=tf.nn.tanh, name='fraternal')
        
        self.tmp_layer = tf.layers.Dense(units=dim_hidden, activation=tf.nn.tanh, name='tmp')
        self.reset_layer = tf.layers.Dense(units=dim_hidden, activation=tf.nn.sigmoid, name='reset')
        self.update_layer = tf.layers.Dense(units=dim_hidden, activation=tf.nn.sigmoid, name='update')            
        
        self.output_layer = output_layer
        
    def __call__(self, state_ancestral, state_fraternal, reuse=True):
        with tf.variable_scope('input', reuse=reuse):
            state_ancestral = self.ancestral_layer(state_ancestral)
            state_fraternal = self.fraternal_layer(state_fraternal)
            
        with tf.variable_scope('output', reuse=reuse):
            state_hidden = state_ancestral + state_fraternal
            
            gate_reset = self.reset_layer(state_hidden)
            gate_update = self.update_layer(state_hidden)
            
            state_reset = tf.multiply(gate_reset, state_hidden)
            state_tmp = self.tmp_layer(state_reset)
            state_update = tf.multiply(gate_update, state_hidden) + tf.multiply((tf.constant(1., dtype=tf.float32)-gate_update), state_tmp)
            
            if self.output_layer is not None: 
                output = self.output_layer(state_update)
            else:
                output = state_update
            
        return output, state_update
    
    def get_initial_state(self, name):
        initial_state = tf.get_variable(name, [1, self.dim_hidden], dtype=tf.float32)
        return initial_state
    
    def get_zero_state(self, name):
        zero_state = tf.zeros([1, self.dim_hidden], dtype=tf.float32, name=name)
        return zero_state    


def doubly_rnn(dim_hidden, tree_idxs, initial_state_parent=None, initial_state_sibling=None, output_layer=None, name='', reuse=False):
    outputs, states_parent = {}, {}
    
    with tf.variable_scope(name, reuse=reuse):
        doubly_rnn_cell = DoublyGRUCell(dim_hidden, output_layer)

        if initial_state_parent is None: 
            initial_state_parent = doubly_rnn_cell.get_initial_state('init_state_parent')
        if initial_state_sibling is None: 
            initial_state_sibling = doubly_rnn_cell.get_zero_state('init_state_sibling')
            
        output, state_sibling = doubly_rnn_cell(initial_state_parent, initial_state_sibling, reuse=False)
        outputs[0], states_parent[0] = output, state_sibling

        for parent_idx, child_idxs in tree_idxs.items():
            state_parent = states_parent[parent_idx]
            state_sibling = initial_state_sibling
            for child_idx in child_idxs:
                output, state_sibling = doubly_rnn_cell(state_parent, state_sibling)
                outputs[child_idx], states_parent[child_idx] = output, state_sibling
                
    states_hidden = states_parent
    return outputs, states_hidden

def nhdp(tree_sticks_path, tree_sticks_depth, tree_idxs):
    tree_prob_path = {}
    tree_rest_prob_depth = {}
    tree_prob_topic = {}
    # calculate topic probability and save
    tree_prob_path[0] = 1.
    tree_rest_prob_depth[0] = 1. - tree_sticks_depth[0]
    tree_prob_topic[0] = tree_prob_path[0] * tree_sticks_depth[0]

    for parent_idx, child_idxs in tree_idxs.items():
        rest_prob_path = tree_prob_path[parent_idx]
        for child_idx in child_idxs:
            stick_path = tree_sticks_path[child_idx]
            if child_idx == child_idxs[-1]:
                prob_path = rest_prob_path * 1.
            else:
                prob_path = rest_prob_path * stick_path

            tree_prob_path[child_idx] = prob_path
            rest_prob_path -= prob_path
            
            if not child_idx in tree_idxs: # leaf childs
                tree_prob_topic[child_idx] = tree_prob_path[child_idx] * tree_rest_prob_depth[parent_idx] * 1.
            else:
                tree_prob_topic[child_idx] = tree_prob_path[child_idx] * tree_rest_prob_depth[parent_idx] * tree_sticks_depth[child_idx]
                tree_rest_prob_depth[child_idx] = tree_rest_prob_depth[parent_idx] * (1-tree_sticks_depth[child_idx])
            
    return tree_prob_topic

def get_prob_topic(tree_prob_leaf, prob_depth, config):
    tree_prob_topic = defaultdict(float)
    leaf_ancestor_idxs = {leaf_idx: get_ancestor_idxs(leaf_idx, config.child_to_parent_idxs) for leaf_idx in tree_prob_leaf}
    for leaf_idx, ancestor_idxs in leaf_ancestor_idxs.items():
        prob_leaf = tree_prob_leaf[leaf_idx]
        for i, ancestor_idx in enumerate(ancestor_idxs):
            prob_ancestor = prob_leaf * tf.expand_dims(prob_depth[:, i], -1)
            tree_prob_topic[ancestor_idx] += prob_ancestor
    prob_topic = tf.concat([tree_prob_topic[topic_idx] for topic_idx in config.topic_idxs], -1)
    return prob_topic
