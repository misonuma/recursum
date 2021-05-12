#coding: utf-8
from collections import defaultdict
import numpy as np
import tensorflow as tf
import pdb

from tree import get_ancestor_idxs

class DoublyRNNCell:
    def __init__(self, dim_hidden, output_layer=None, dropout_layer=None, layer_norm=None):
        self.dim_hidden = dim_hidden
        
        self.ancestral_layer=tf.layers.Dense(units=dim_hidden, activation=tf.nn.tanh, name='ancestral')
        self.fraternal_layer=tf.layers.Dense(units=dim_hidden, activation=tf.nn.tanh, name='fraternal')
        self.hidden_layer = tf.layers.Dense(units=dim_hidden, name='hidden')
        
        self.output_layer=output_layer
        
    def __call__(self, state_ancestral, state_fraternal, reuse=True):
        with tf.variable_scope('input', reuse=reuse):
            state_ancestral = self.ancestral_layer(state_ancestral)
            state_fraternal = self.fraternal_layer(state_fraternal)

        with tf.variable_scope('output', reuse=reuse):
            state_hidden = self.hidden_layer(state_ancestral + state_fraternal)
            if self.output_layer is not None: 
                output = self.output_layer(state_hidden)
            else:
                output = state_hidden
            
        return output, state_hidden
    
    def get_initial_state(self, name):
        initial_state = tf.get_variable(name, [1, self.dim_hidden], dtype=tf.float32)
        return initial_state
    
    def get_zero_state(self, name):
        zero_state = tf.zeros([1, self.dim_hidden], dtype=tf.float32, name=name)
        return zero_state    

class DoublyGRUCell:
    def __init__(self, dim_hidden, output_layer=None, dropout_layer=None, layer_norm=None, sigmoid=False):
        self.dim_hidden = dim_hidden
        
        self.ancestral_layer = tf.layers.Dense(units=dim_hidden, activation=tf.nn.tanh, name='ancestral')
        self.fraternal_layer = tf.layers.Dense(units=dim_hidden, activation=tf.nn.tanh, name='fraternal')
        
        self.tmp_layer = tf.layers.Dense(units=dim_hidden, activation=tf.nn.tanh, name='tmp')
        if sigmoid:
            self.reset_layer = tf.layers.Dense(units=dim_hidden, activation=tf.nn.sigmoid, name='reset')
            self.update_layer = tf.layers.Dense(units=dim_hidden, activation=tf.nn.sigmoid, name='update')            
        else:
            self.reset_layer = tf.layers.Dense(units=dim_hidden, activation=tf.nn.tanh, name='reset')
            self.update_layer = tf.layers.Dense(units=dim_hidden, activation=tf.nn.tanh, name='update')
        
        self.output_layer = output_layer
        self.dropout_layer = dropout_layer
        self.layer_norm = layer_norm
        
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
            
            if self.layer_norm is not None: state_update = self.layer_norm(state_update)
            if self.dropout_layer is not None: state_update = self.dropout_layer(state_update)
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


def doubly_rnn(dim_hidden, tree_idxs, initial_state_parent=None, initial_state_sibling=None, output_layer=None, dropout_layer=None, layer_norm=None, sigmoid=False, cell='rnn', name='', reuse=False):
    outputs, states_parent = {}, {}
    
    with tf.variable_scope(name, reuse=reuse):
        if cell=='rnn':
            doubly_rnn_cell = DoublyRNNCell(dim_hidden, output_layer, dropout_layer, layer_norm, sigmoid)
        elif cell=='gru':
            doubly_rnn_cell = DoublyGRUCell(dim_hidden, output_layer, dropout_layer, layer_norm, sigmoid)
        elif cell=='lstm':
            doubly_rnn_cell = DoublyLSTMCell(dim_hidden, output_layer, dropout_layer, layer_norm, sigmoid)

        if initial_state_parent is None: 
            initial_state_parent = doubly_rnn_cell.get_initial_state('init_state_parent')
        if initial_state_sibling is None: 
            initial_state_sibling = doubly_rnn_cell.get_zero_state('init_state_sibling')
            
        if cell=='lstm':
            initial_state_memory_parent = doubly_rnn_cell.get_zero_state('init_memory_state_parent')
            initial_state_parent = tf.contrib.rnn.LSTMStateTuple(initial_state_memory_parent, initial_state_parent)
            initial_state_memory_sibling = doubly_rnn_cell.get_zero_state('init_memory_state_sibling')
            initial_state_sibling = tf.contrib.rnn.LSTMStateTuple(initial_state_memory_sibling, initial_state_sibling)
            
        output, state_sibling = doubly_rnn_cell(initial_state_parent, initial_state_sibling, reuse=False)
        outputs[0], states_parent[0] = output, state_sibling

        for parent_idx, child_idxs in tree_idxs.items():
            state_parent = states_parent[parent_idx]
            state_sibling = initial_state_sibling
            for child_idx in child_idxs:
                output, state_sibling = doubly_rnn_cell(state_parent, state_sibling)
                outputs[child_idx], states_parent[child_idx] = output, state_sibling
                
    if cell=='lstm':
        states_hidden = {topic_idx: state_parent.h for topic_idx, state_parent in states_parent.items()}
    else:
        states_hidden = states_parent

    return outputs, states_hidden

class RNNCell:
    def __init__(self, dim_hidden, output_layer=None):
        self.dim_hidden = dim_hidden
        self.hidden_layer = tf.layers.Dense(units=dim_hidden, activation=tf.nn.tanh, name='hidden')
        self.output_layer=output_layer
        
    def __call__(self, state, reuse=True):
        with tf.variable_scope('output', reuse=reuse):
            state_hidden = self.hidden_layer(state)
            if self.output_layer is not None: 
                output = self.output_layer(state_hidden)
            else:
                output = state_hidden
            
        return output, state_hidden
    
    def get_initial_state(self, name):
        initial_state = tf.get_variable(name, [1, self.dim_hidden], dtype=tf.float32)
        return initial_state
    
    def get_zero_state(self, name):
        zero_state = tf.zeros([1, self.dim_hidden], dtype=tf.float32, name=name)
        return zero_state

class GRUCell:
    def __init__(self, dim_hidden, output_layer=None):
        self.dim_hidden = dim_hidden

        self.reset_layer=tf.layers.Dense(units=dim_hidden, activation=tf.nn.tanh, name='reset')
        self.tmp_layer=tf.layers.Dense(units=dim_hidden, activation=tf.nn.tanh, name='tmp')
        self.update_layer=tf.layers.Dense(units=dim_hidden, activation=tf.nn.tanh, name='update')   
        
        self.output_layer=output_layer
        
    def __call__(self, state_hidden, reuse=True):
        with tf.variable_scope('output', reuse=reuse):
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

def rnn(dim_hidden, max_depth, initial_state=None, output_layer=None, cell='rnn', name='', concat=True):
    outputs, states_hidden = [], []
    with tf.variable_scope(name, reuse=False):
        if cell=='rnn':
            rnn_cell = RNNCell(dim_hidden, output_layer)
        elif cell=='gru':
            rnn_cell = GRUCell(dim_hidden, output_layer)
        elif cell=='lstm':
            rnn_cell = LSTMCell(dim_hidden, output_layer)

        if initial_state is not None: 
            state_hidden = initial_state
        else:
            state_hidden = rnn_cell.get_initial_state('init_state')
            
        if cell=='lstm':
            state_memory = rnn_cell.get_zero_state('init_memory_state')
            state_hidden = tf.contrib.rnn.LSTMStateTuple(state_memory, state_hidden)
        
        for depth in range(max_depth):
            if depth == 0:                
                output, state_hidden = rnn_cell(state_hidden, reuse=False)
            else:
                output, state_hidden = rnn_cell(state_hidden, reuse=True)
            outputs.append(output)

            if cell=='lstm':
                states_hidden.append(state_hidden[-1])
            else:
                states_hidden.append(state_hidden)

    outputs = tf.concat(outputs, 1) if concat else tf.concat(outputs, 0)
    states_hidden = tf.concat(states_hidden, 0)
    return outputs, states_hidden

def tsbp(tree_sticks_topic, tree_idxs):
    tree_prob_topic = {}
    tree_prob_leaf = {}
    # calculate topic probability and save
    tree_prob_topic[0] = 1.

    for parent_idx, child_idxs in tree_idxs.items():
        rest_prob_topic = tree_prob_topic[parent_idx]
        for child_idx in child_idxs:
            stick_topic = tree_sticks_topic[child_idx]
            if child_idx == child_idxs[-1]:
                prob_topic = rest_prob_topic * 1.
            else:
                prob_topic = rest_prob_topic * stick_topic

            if not child_idx in tree_idxs: # leaf childs
                tree_prob_leaf[child_idx] = prob_topic
            else:
                tree_prob_topic[child_idx] = prob_topic

            rest_prob_topic -= prob_topic
    return tree_prob_leaf

def sbp(sticks_depth, max_depth):
    prob_depth_list = []
    rest_prob_depth = 1.
    for depth in range(max_depth):
        stick_depth = tf.expand_dims(sticks_depth[:, depth], 1)
        if depth == max_depth -1:
            prob_depth = rest_prob_depth * 1.
        else:
            prob_depth = rest_prob_depth * stick_depth
        prob_depth_list.append(prob_depth)
        rest_prob_depth -= prob_depth

    prob_depth = tf.concat(prob_depth_list, 1)
    return prob_depth

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

def rcrp_tmp(tree_sent_logits, tree_idxs, depth_topic_idxs):
    sorted_depth_topic_idxs = sorted([(depth, topic_idxs) for depth, topic_idxs in depth_topic_idxs.items()], key=lambda t: t[0], reverse=True)
    recur_sent_logits = {}
    for sorted_depth, topic_idxs in sorted_depth_topic_idxs:
        if sorted_depth == max(depth_topic_idxs.keys()): # if leaf
            for topic_idx in topic_idxs:
                recur_sent_logits[topic_idx] = tree_sent_logits[topic_idx]
        else: # if non-leaf
            for topic_idx in topic_idxs:
                recur_sent_logits[topic_idx] = tf.reduce_sum(tf.concat([recur_sent_logits[child_idx] for child_idx in tree_idxs[topic_idx]] \
                                                                       + [tree_sent_logits[topic_idx]], -1), -1, keepdims=True)
                
    return recur_sent_logits

def rcrp(recur_probs_sent_topic_posterior, tree_idxs, topic_idxs, depth_topic_idxs):
    sorted_depth_topic_idxs = sorted([(depth, topic_idxs) for depth, topic_idxs in depth_topic_idxs.items()], key=lambda t: t[0], reverse=True)
    tree_probs_sent_topic_posterior = {}
    for sorted_depth, parent_idxs in sorted_depth_topic_idxs:
        if sorted_depth == max(depth_topic_idxs.keys()): # if leaf
            for parent_idx in parent_idxs:
                tree_probs_sent_topic_posterior[parent_idx] = tf.expand_dims(recur_probs_sent_topic_posterior[:, :, topic_idxs.index(parent_idx)], -1)
        else: # if non-leaf
            for parent_idx in parent_idxs:
                tree_probs_sent_topic_posterior[parent_idx] = \
                    tf.reduce_sum(tf.concat([tree_probs_sent_topic_posterior[child_idx] for child_idx in tree_idxs[parent_idx]] + \
                                        [tf.expand_dims(recur_probs_sent_topic_posterior[:, :, topic_idxs.index(parent_idx)], -1)], -1), -1, keepdims=True)
                
    return tree_probs_sent_topic_posterior

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
