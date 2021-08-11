#coding:utf-8
import pdb

import numpy as np
import tensorflow as tf

from components import sample_latents
from nn import doubly_rnn, nhdp, rcrp
from topic_beam_search_decoder import BeamSearchDecoder
from basic_decoder import BasicDecoder
from topic_helper import TrainingHelper, GumbelSoftmaxEmbeddingHelper, SampleEmbeddingHelper
from attention_wrapper import LuongAttention, AttentionWrapper
from decoder import dynamic_decode

def get_embeddings(model):
    with tf.variable_scope('word', reuse=False):
        pad_embedding = tf.zeros([1, model.config.dim_emb], dtype=tf.float32)
        unk_embedding = tf.get_variable('unk', [1, model.config.dim_emb], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        other_embeddings = tf.get_variable('emb', [model.config.n_vocab-2, model.config.dim_emb], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        embeddings = tf.concat([pad_embedding, unk_embedding, other_embeddings], 0) # n_vocab x dim_emb
    return embeddings

def encode_inputs(model, enc_inputs, sent_l, cell_name='', reuse=False):
#     enc_inputs_flat = tf.reshape(enc_inputs, [model.batch_l*model.max_doc_l, model.max_sent_l, model.config.dim_emb])
    enc_inputs_flat = tf.reshape(enc_inputs, [tf.shape(enc_inputs)[0]*tf.shape(enc_inputs)[1], tf.shape(enc_inputs)[2], enc_inputs.get_shape()[-1]]) # batch_l x max_doc_l x max_sent_l x dim_emb
    sent_l_flat = tf.reshape(sent_l, [tf.shape(sent_l)[0]*tf.shape(sent_l)[1]])
    flat_l = tf.shape(enc_inputs_flat)[0]
    
    with tf.variable_scope(cell_name + 'fw', initializer=tf.contrib.layers.xavier_initializer(), dtype = tf.float32, reuse=reuse):
        if model.config.cell == 'gru':
            fw_cell = tf.contrib.rnn.GRUCell(model.config.dim_hidden)
            fw_cell = tf.contrib.rnn.DropoutWrapper(fw_cell, output_keep_prob=model.enc_keep_prob)
        elif model.config.cell == 'lstm':
            fw_cell = tf.contrib.rnn.LSTMCell(model.config.dim_hidden)
            fw_cell = tf.contrib.rnn.DropoutWrapper(fw_cell, output_keep_prob=model.enc_keep_prob)
        fw_initial_state = fw_cell.zero_state(flat_l, tf.float32)
    
    with tf.variable_scope(cell_name + 'bw', initializer=tf.contrib.layers.xavier_initializer(), dtype = tf.float32, reuse=reuse):
        if model.config.cell == 'gru':
            bw_cell = tf.contrib.rnn.GRUCell(model.config.dim_hidden)
            bw_cell = tf.contrib.rnn.DropoutWrapper(bw_cell, output_keep_prob=model.enc_keep_prob)
        elif model.config.cell == 'lstm':
            bw_cell = tf.contrib.rnn.LSTMCell(model.config.dim_hidden)
            bw_cell = tf.contrib.rnn.DropoutWrapper(bw_cell, output_keep_prob=model.enc_keep_prob)            
        bw_initial_state = bw_cell.zero_state(flat_l, tf.float32)

    bi_outputs_flat, bi_output_state_flat = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, enc_inputs_flat,
                                                                                                                                     initial_state_fw=fw_initial_state,
                                                                                                                                     initial_state_bw=bw_initial_state,
                                                                                                                                     sequence_length=sent_l_flat)
    
    if model.config.cell == 'gru':
        outputs_flat = tf.concat(list(bi_outputs_flat), -1)
        output_state_flat = tf.concat(list(bi_output_state_flat), -1)
    elif model.config.cell=='lstm':
        output_state_flat = tf.concat([state_flat.h for state_flat in bi_output_state_flat], 1)
    
    outputs_ = tf.reshape(outputs_flat, [tf.shape(enc_inputs)[0], tf.shape(enc_inputs)[1], tf.shape(enc_inputs)[2], model.config.dim_hidden*2]) # batch_l x max_doc_l x max_sent_l x dim_hidden
    outputs = tf.layers.Dense(units=model.config.dim_hidden, activation=tf.nn.tanh, name='outputs')(outputs_)
    
    output_state_ = tf.reshape(output_state_flat, [tf.shape(enc_inputs)[0], tf.shape(enc_inputs)[1], model.config.dim_hidden*2]) # batch_l x max_doc_l x dim_hidden    
    output_state = tf.layers.Dense(units=model.config.dim_hidden, activation=tf.nn.tanh, name='output_state')(output_state_)
    return outputs, output_state

def encode_latents_gauss(input_state, dim_latent, sample, config, name, min_logvar=None, max_logvar=None):
    means = tf.layers.Dense(units=dim_latent, name='means_'+name)(input_state) # batch_l x max_doc_l x dim_latent
    logvars = tf.layers.Dense(units=dim_latent, kernel_initializer=tf.constant_initializer(0), bias_initializer=tf.constant_initializer(0), name='logvars_'+name)(input_state) # batch_l x max_doc_l x dim_latent
    
    if min_logvar is not None and max_logvar is not None: assert max_logvar > min_logvar
    if min_logvar is not None: logvars = tf.maximum(logvars, min_logvar)
    if max_logvar is not None: logvars = tf.minimum(logvars, max_logvar)
        
    latents = sample_latents(means, logvars, seed=config.seed, sample=sample)
    return latents, means, logvars

def encode_gsm_probs_topic_posterior(model, dim_hidden, latents_probs_sent_topic_posterior, mask_doc, config):
    # sample topic prob. dist. for each sentence
    probs_sent_topic_posterior_ = tf.layers.Dense(units=model.config.n_topic, activation=tf.nn.softmax, name='prob_topic')(latents_probs_sent_topic_posterior) # batch_l x max_doc_l x n_topic
    probs_sent_topic_posterior = tf.maximum(probs_sent_topic_posterior_, config.CLIP)
    
    return probs_sent_topic_posterior, None, None

def encode_nhdp_probs_topic_posterior(model, dim_hidden, latents_probs_sent_topic_posterior, mask_doc, config):
    # sample topic prob. dist. for each sentence
    probs_sent_topic_layer = lambda h: tf.nn.sigmoid(tf.tensordot(latents_probs_sent_topic_posterior, h, axes=[[-1], [-1]]))
    
    if model.config.renew_drnn:
        dropout_layer = None
        layer_norm = None
        sigmoid = True
    else:
        dropout_layer = tf.layers.Dropout(model.t_variables['enc_keep_prob']) if model.config.dropout_drnn else None # TODO
        layer_norm = tf.contrib.layers.layer_norm if model.config.layernorm_drnn else None # TODO
        sigmoid = False
    
    tree_sent_sticks_path, _ = doubly_rnn(dim_hidden, config.tree_idxs, output_layer=probs_sent_topic_layer, \
                                          dropout_layer=dropout_layer, layer_norm=layer_norm, sigmoid=sigmoid, cell=config.cell, name='sticks_path')
    tree_sent_sticks_depth, _ = doubly_rnn(dim_hidden, config.tree_idxs, output_layer=probs_sent_topic_layer, \
                                          dropout_layer=dropout_layer, layer_norm=layer_norm, sigmoid=sigmoid, cell=config.cell, name='sticks_depth')
    tree_probs_sent_topic_posterior = nhdp(tree_sent_sticks_path, tree_sent_sticks_depth, config.tree_idxs)
    probs_sent_topic_posterior_ = tf.multiply(tf.concat([tree_probs_sent_topic_posterior[topic_idx] for topic_idx in config.topic_idxs], -1), \
                                              tf.expand_dims(mask_doc, -1)) # batch_l x max_doc_l x n_topic
    probs_sent_topic_posterior = tf.maximum(probs_sent_topic_posterior_, config.CLIP)
    
    return probs_sent_topic_posterior, tree_sent_sticks_path, tree_sent_sticks_depth

def decode_output_logits_flat(model, dec_cell, dec_initial_state, dec_inputs, dec_sent_l, latents_input=None):
    dec_initial_state_flat = tf.reshape(dec_initial_state, [model.batch_l*model.max_doc_l, model.config.dim_hidden])
    if model.config.attention:
        dec_initial_state_flat = dec_cell.zero_state(tf.shape(dec_initial_state_flat)[0], dtype=tf.float32).clone(cell_state=dec_initial_state_flat)    
    
    dec_inputs_flat = tf.reshape(dec_inputs, [model.batch_l*model.max_doc_l, model.max_dec_sent_l, model.config.dim_emb]) 
    dec_sent_l_flat = tf.reshape(dec_sent_l, [model.batch_l*model.max_doc_l])

    if latents_input is not None:
        latents_input_flat = tf.reshape(latents_input, [model.batch_l*model.max_doc_l, latents_input.get_shape()[-1]]) # batch_l*max_doc_l x n_topic
        latents_input_flat_tiled = tf.tile(tf.expand_dims(latents_input_flat, 1), [1, model.max_dec_sent_l, 1]) # batch_l*max_doc_l x max_dec_sent_l x n_topic
        dec_inputs_flat = tf.concat([dec_inputs_flat, latents_input_flat_tiled], -1)
        
    memory_idxs_flat = tile_memory_idxs(model, model.memory_idxs, n_tiled=model.max_doc_l)
    
    helper = TrainingHelper(inputs=dec_inputs_flat, sequence_length=dec_sent_l_flat)
    train_decoder = BasicDecoder(cell=dec_cell, helper=helper, initial_state=dec_initial_state_flat, \
                                                         output_layer=model.output_layer, pointer_layer=model.pointer_layer, oov=model.config.oov, memory_idxs=memory_idxs_flat)

    dec_outputs, dec_final_state, _ = dynamic_decode(train_decoder, maximum_iterations=None, \
                                                                    parallel_iterations=model.config.parallel_iterations, swap_memory=model.config.swap_memory)

    return dec_outputs, dec_final_state, dec_sent_l_flat


def decode_output_sample_flat(model, dec_cell, dec_initial_state, softmax_temperature, sample, latents_input=None):
    dec_initial_state_flat = tf.reshape(dec_initial_state, [tf.shape(dec_initial_state)[0]*tf.shape(dec_initial_state)[1], tf.shape(dec_initial_state)[-1]])
    start_tokens = tf.fill([tf.shape(dec_initial_state_flat)[0]], model.config.BOS_IDX)
    end_token = model.config.EOS_IDX
    if model.config.attention:
        dec_initial_state_flat = dec_cell.zero_state(tf.shape(dec_initial_state_flat)[0], dtype=tf.float32).clone(cell_state=dec_initial_state_flat)
    
    if latents_input is not None:
        latents_input_flat = tf.reshape(latents_input, [tf.shape(latents_input)[0]*tf.shape(latents_input)[1], latents_input.get_shape()[-1]])
    
    memory_idxs_flat = tile_memory_idxs(model, model.memory_idxs, n_tiled=tf.shape(dec_initial_state)[1])
    
    helper = GumbelSoftmaxEmbeddingHelper(model.dec_embeddings, start_tokens, end_token, 
                        softmax_temperature=softmax_temperature, seed=model.config.seed, sample=sample, latents_input=latents_input_flat)
    train_decoder = BasicDecoder(cell=dec_cell, helper=helper, initial_state=dec_initial_state_flat, \
                                                         output_layer=model.output_layer, pointer_layer=model.pointer_layer, oov=model.config.oov, memory_idxs=memory_idxs_flat)

    maximum_iterations = model.config.maximum_iterations
    dec_outputs, dec_final_state, output_sent_l_flat = dynamic_decode(train_decoder, maximum_iterations=maximum_iterations, \
                                                                                                    parallel_iterations=model.config.parallel_iterations, swap_memory=model.config.swap_memory)
    return dec_outputs, dec_final_state, output_sent_l_flat

def decode_beam_output_token_idxs(model, beam_dec_cell, dec_initial_state, latents_input=None, name=None):
    dec_initial_state_flat = tf.reshape(dec_initial_state, [tf.shape(dec_initial_state)[0]*tf.shape(dec_initial_state)[1], tf.shape(dec_initial_state)[-1]])
    beam_dec_initial_state_flat = tf.contrib.seq2seq.tile_batch(dec_initial_state_flat, multiplier=model.config.beam_width)
    
    if model.config.attention:
        beam_dec_initial_state_flat = beam_dec_cell.zero_state(tf.shape(beam_dec_initial_state_flat)[0], dtype=tf.float32).clone(cell_state=beam_dec_initial_state_flat)
        
    if latents_input is not None:
        latents_input_flat = tf.reshape(latents_input, [tf.shape(latents_input)[0]*tf.shape(latents_input)[1], tf.shape(latents_input)[-1]])
        beam_latents_input_flat = tf.contrib.seq2seq.tile_batch(latents_input_flat, multiplier=model.config.beam_width)
    
    start_tokens = tf.fill([tf.shape(dec_initial_state_flat)[0]], model.config.BOS_IDX)
    end_token = model.config.EOS_IDX
    
    memory_idxs_flat = tile_memory_idxs(model, model.memory_idxs, n_tiled=tf.shape(dec_initial_state)[1], beam_width=model.config.beam_width)

    beam_decoder = BeamSearchDecoder(
        cell=beam_dec_cell,
        embedding=model.dec_embeddings,
        start_tokens=start_tokens,
        end_token=end_token,
        initial_state=beam_dec_initial_state_flat,
        beam_width=model.config.beam_width,
        output_layer=model.output_layer,
        pointer_layer=model.pointer_layer,
        memory_idxs=memory_idxs_flat,
        length_penalty_weight=model.config.length_penalty_weight,
        latents_input=beam_latents_input_flat)

    beam_dec_outputs, _, beam_output_sent_l_flat = dynamic_decode(beam_decoder, \
                                                                                                maximum_iterations=model.config.maximum_iterations, \
                                                                                                parallel_iterations=model.config.parallel_iterations, swap_memory=model.config.swap_memory)
    beam_output_token_idxs_flat = beam_dec_outputs.predicted_ids[:, :, 0]
    beam_output_token_idxs = tf.reshape(beam_output_token_idxs_flat, \
                [tf.shape(dec_initial_state)[0], tf.shape(dec_initial_state)[1], tf.shape(beam_output_token_idxs_flat)[-1]], name=name)
    return beam_output_token_idxs


def decode_sample_output_token_idxs(model, dec_cell, dec_initial_state, latents_input=None, name=None):
    dec_initial_state_flat = tf.reshape(dec_initial_state, [tf.shape(dec_initial_state)[0]*tf.shape(dec_initial_state)[1], tf.shape(dec_initial_state)[-1]])
    start_tokens = tf.fill([tf.shape(dec_initial_state_flat)[0]], model.config.BOS_IDX)
    end_token = model.config.EOS_IDX
    if model.config.attention:
        dec_initial_state_flat = dec_cell.zero_state(tf.shape(dec_initial_state_flat)[0], dtype=tf.float32).clone(cell_state=dec_initial_state_flat)
    
    if latents_input is not None:
        latents_input_flat = tf.reshape(latents_input, [tf.shape(latents_input)[0]*tf.shape(latents_input)[1], latents_input.get_shape()[-1]])
    
    memory_idxs_flat = tile_memory_idxs(model, model.memory_idxs, n_tiled=tf.shape(dec_initial_state)[1])
    
    helper = SampleEmbeddingHelper(model.dec_embeddings, start_tokens, end_token, \
                                   nucleus=model.config.nucleus, softmax_temperature=None, seed=model.config.seed, latents_input=latents_input_flat)
    sample_decoder = BasicDecoder(cell=dec_cell, helper=helper, initial_state=dec_initial_state_flat, \
                                                         output_layer=model.output_layer, pointer_layer=model.pointer_layer, oov=model.config.oov, memory_idxs=memory_idxs_flat)

    maximum_iterations = model.config.maximum_iterations
    sample_dec_outputs, _, _ = dynamic_decode(sample_decoder, maximum_iterations=maximum_iterations, \
                                                                                                    parallel_iterations=model.config.parallel_iterations, swap_memory=model.config.swap_memory)
    
    sample_output_token_idxs_flat = sample_dec_outputs.sample_id
    sample_output_token_idxs = tf.reshape(sample_output_token_idxs_flat, \
                              [tf.shape(dec_initial_state)[0], tf.shape(dec_initial_state)[1], tf.shape(sample_output_token_idxs_flat)[-1]], name=name)
    
    return sample_output_token_idxs


def wrap_attention(model, dec_cell, sent_outputs, n_tiled, beam_width=None):
    tiled_sent_outputs = tf.tile(tf.expand_dims(sent_outputs, 1), [1, n_tiled, 1, 1, 1]) # batch_l x max_doc_l x max_doc_l x max_sent_l x dim_hidden
    tiled_sent_outputs_flat = tf.reshape(tiled_sent_outputs, [model.batch_l*n_tiled, model.max_doc_l*model.max_sent_l, tiled_sent_outputs.get_shape()[-1]])
    tiled_mask_sent = tf.sequence_mask(tf.tile(tf.expand_dims(model.sent_l, 1), [1, n_tiled, 1])) # batch_l x max_doc_l x max_doc_l x max_sent_l
    tiled_mask_sent_flat = tf.reshape(tiled_mask_sent, [model.batch_l*n_tiled, model.max_doc_l*model.max_sent_l]) # batch_l*max_doc_l x max_doc_l*max_sent_l
    
    if beam_width:
        tiled_sent_outputs_flat = tf.contrib.seq2seq.tile_batch(tiled_sent_outputs_flat, multiplier=beam_width)
        tiled_mask_sent_flat = tf.contrib.seq2seq.tile_batch(tiled_mask_sent_flat, multiplier=beam_width)
    
    att_mechanism = LuongAttention(num_units=model.config.dim_hidden, 
                                                                memory=tiled_sent_outputs_flat, \
                                                                score_mask=tiled_mask_sent_flat,
                                                                coverage=model.config.coverage)

    att_cell = AttentionWrapper(dec_cell,
                                                      attention_mechanism=att_mechanism,
                                                      attention_layer_size=model.config.dim_emb,
                                                      alignment_history=(model.config.coverage or model.config.regularizer),
#                                                       alignment_history=True,
                                                      prob_gen_layer=model.prob_gen_layer)

    return att_cell


def tile_memory_idxs(model, enc_input_idxs, n_tiled, beam_width=None):
    tiled_enc_input_idxs = tf.tile(tf.expand_dims(enc_input_idxs, 1), [1, n_tiled, 1, 1])
    tiled_enc_input_idxs_flat = tf.reshape(tiled_enc_input_idxs, [model.batch_l*n_tiled, model.max_doc_l*model.max_sent_l])
    if beam_width:
        tiled_enc_input_idxs_flat = tf.contrib.seq2seq.tile_batch(tiled_enc_input_idxs_flat, multiplier=beam_width)
    return tiled_enc_input_idxs_flat


