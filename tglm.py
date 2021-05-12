#coding: utf-8
import pdb
import random
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

from components import tf_log, tf_clip_vals, tf_clip_means, tf_clip_covs, sample_latents, sample_latents_fullcov, sample_gumbels, get_params_topic_prior, compute_kl_losses, compute_kl_losses_sent_gauss, compute_kl_losses_topic_gauss, compute_kl_losses_topic_paris_gauss
from seq2seq import encode_inputs, encode_latents_gauss, encode_gsm_probs_topic_posterior, decode_output_logits_flat, decode_output_sample_flat, decode_beam_output_token_idxs
from attention_wrapper import HierarchicalAttention, AttentionWrapper
from nn import doubly_rnn, rnn, tsbp, sbp, nhdp, get_prob_topic

class FlatTopicGuidedLanguageModel():
    def __init__(self, config):
        tf.reset_default_graph()
        
        np.random.seed(config.seed)
        random.seed(config.seed)
        tf.set_random_seed(config.seed)
        
        self.config = config
        
        t_variables = {}
        t_variables['batch_l'] = tf.placeholder(tf.int32, name='batch_l')
        t_variables['doc_l'] = tf.placeholder(tf.int32, [None], name='doc_l') # batch_l
        t_variables['sent_l'] = tf.placeholder(tf.int32, [None, None], name='sent_l') # batch_l x max_doc_l
        t_variables['dec_sent_l'] = tf.placeholder(tf.int32, [None, None], name='sent_l') # batch_l x max_doc_l
        t_variables['bows'] = tf.placeholder(tf.float32, [None, None, config.dim_bow], name='bow') # batch_l x max_doc_l x dim_bow
        t_variables['doc_bows'] = tf.placeholder(tf.float32, [None, config.dim_bow], name='doc_bow') # batch_l x dim_bow
        t_variables['enc_input_idxs'] = tf.placeholder(tf.int32, [None, None, None], name='enc_input_idxs') # batch_l x max_doc_l x max_sent_l
        t_variables['dec_input_idxs'] = tf.placeholder(tf.int32, [None, None, None], name='dec_input_idxs') # batch_l x max_doc_l x max_sent_l+1
        t_variables['dec_target_idxs'] = tf.placeholder(tf.int32, [None, None, None], name='dec_target_idxs') # batch_l x max_doc_l x max_sent_l+1
        t_variables['enc_keep_prob'] = tf.placeholder(tf.float32, name='enc_keep_prob')
        t_variables['dec_keep_prob'] = tf.placeholder(tf.float32, name='dec_keep_prob')
        t_variables['sample'] = tf.placeholder(tf.bool, name='sample')
        self.t_variables = t_variables
        
        self.build()
        
    def build(self):
        self.global_step = tf.Variable(0, name='global_step',trainable=False)
        self.softmax_temperature = tf.maximum( \
                                              self.config.max_temperature-tf.cast(tf.divide(self.global_step, tf.constant(self.config.linear_steps)), dtype=tf.float32), \
                                              self.config.min_temperature)
        
        with tf.name_scope('t_variables'):
            self.sample = self.t_variables['sample']
            
            self.batch_l = self.t_variables['batch_l']
            self.doc_l = self.t_variables['doc_l']
            self.sent_l = self.t_variables['sent_l']
            self.dec_sent_l = self.t_variables['dec_sent_l'] # batch_l x max_doc_l

            self.max_doc_l = tf.reduce_max(self.doc_l)
            self.max_sent_l = tf.reduce_max(self.sent_l)
            self.max_dec_sent_l = tf.reduce_max(self.dec_sent_l) # = max_sent_l + 1

            self.mask_doc = tf.sequence_mask(self.doc_l, dtype=tf.float32)
            self.mask_sent = tf.sequence_mask(self.sent_l, dtype=tf.float32)

            mask_bow = np.zeros(self.config.n_vocab)
            mask_bow[self.config.bow_idxs] = 1.
            self.mask_bow = tf.constant(mask_bow, dtype=tf.float32)
            
            self.enc_keep_prob = self.t_variables['enc_keep_prob']
        
        # ------------------------------Encoder ------------------------------        
        with tf.variable_scope('emb'):
            with tf.variable_scope('word', reuse=False):
                pad_embedding = tf.zeros([1, self.config.dim_emb], dtype=tf.float32)
                nonpad_embeddings = tf.get_variable('emb', [self.config.n_vocab-1, self.config.dim_emb], dtype=tf.float32, \
                                                                initializer=tf.contrib.layers.xavier_initializer())
                self.embeddings = tf.concat([pad_embedding, nonpad_embeddings], 0) # n_vocab x dim_emb
                self.bow_embeddings = tf.nn.embedding_lookup(self.embeddings, self.config.bow_idxs) # dim_bow x dim_emb    

                # get sentence embeddings
                self.enc_input_idxs = tf.one_hot(self.t_variables['enc_input_idxs'], depth=self.config.n_vocab) # batch_l x max_doc_l x max_sent_l x n_vocab
                self.enc_inputs = tf.tensordot(self.enc_input_idxs, self.embeddings, axes=[[-1], [0]]) # batch_l x max_doc_l x max_sent_l x dim_emb
            
            with tf.variable_scope('sent', reuse=False):
                self.sent_outputs, self.sent_state = \
                    encode_inputs(self, enc_inputs=self.enc_inputs, sent_l=self.sent_l) # batch_l x max_doc_l x dim_hidden*2
                    
        with tf.variable_scope('enc'):
            # get sentence latents
            with tf.variable_scope('latents_sent', reuse=False):
                self.w_topic_posterior = tf.get_variable('topic_posterior/kernel', [self.config.n_topic, self.sent_state.shape[-1], self.config.dim_hidden], dtype=tf.float32)
                self.b_topic_posterior = tf.get_variable('topic_posterior/bias', [1, self.config.n_topic, self.config.dim_hidden], dtype=tf.float32)

                self.topic_state = tf.reduce_sum(self.sent_state * tf.expand_dims(self.mask_doc, -1), -2) / tf.reduce_sum(self.mask_doc, -1, keepdims=True)
                self.hidden_topic_posterior = tf.tensordot(self.topic_state, self.w_topic_posterior, axes=[[1], [1]]) + self.b_topic_posterior # batch_l x n_topic x dim_hidden
                                               
        # ------------------------------Discriminator------------------------------        
        with tf.variable_scope('disc'):
            with tf.variable_scope('prob_topic', reuse=False):
                # encode by TSNTM
                self.probs_sent_topic_posterior, _, _ = \
                    encode_gsm_probs_topic_posterior(self, self.hidden_topic_posterior.get_shape()[-1], self.hidden_topic_posterior, self.mask_doc, self.config) # batch_l x max_doc_l x n_topic
                        
            with tf.name_scope('latents_topic'):
                # get topic sentence posterior distribution for each document
                self.probs_topic_posterior = tf.reduce_sum(self.probs_sent_topic_posterior, 1) # batch_l x n_topic

                self.means_sent_topic_posterior = tf.multiply(tf.expand_dims(self.probs_sent_topic_posterior, -1), \
                        tf.expand_dims(self.means_sent_posterior, -2)) # batch_l x max_doc_l x n_topic x dim_latent
                self.means_topic_posterior_ = tf.reduce_sum(self.means_sent_topic_posterior, 1) / \
                        tf.expand_dims(self.probs_topic_posterior, -1) # batch_l x n_topic x dim_latent
                self.means_topic_posterior = tf_clip_means(self.means_topic_posterior_, self.probs_topic_posterior)

                diffs_sent_topic_posterior = tf.expand_dims(self.means_sent_posterior, 2) - \
                        tf.expand_dims(self.means_topic_posterior, 1) # batch_l x max_doc_l x n_topic x dim_latent
                self.covs_sent_topic_posterior = tf.multiply(tf.expand_dims(tf.expand_dims(self.probs_sent_topic_posterior, -1), -1), \
                        tf.matrix_diag(tf.expand_dims(tf.exp(self.logvars_sent_posterior), 2)) + tf.matmul(tf.expand_dims(diffs_sent_topic_posterior, -1), \
                        tf.expand_dims(diffs_sent_topic_posterior, -2))) # batch_l x max_doc_l x n_topic x dim_latent x dim_latent
                self.covs_topic_posterior_ = tf.reduce_sum(self.covs_sent_topic_posterior, 1) / \
                        tf.expand_dims(tf.expand_dims(self.probs_topic_posterior, -1), -1) # batch_l x n_topic x dim_latent x dim_latent
                self.covs_topic_posterior = tf_clip_covs(self.covs_topic_posterior_, self.probs_topic_posterior)
                
                self.latents_topic_posterior = sample_latents_fullcov(self.means_topic_posterior, self.covs_topic_posterior, \
                                                                      seed=self.config.seed, sample=self.sample)

                self.means_topic_prior = tf.zeros([self.batch_l, self.config.n_topic, self.config.dim_latent], dtype=tf.float32) # batch_l x n_topic x dim_latent
                self.covs_topic_prior = tf.eye(self.config.dim_latent, batch_shape=[self.batch_l, self.config.n_topic], dtype=tf.float32) * self.config.cov_root
                
        # ------------------------------Decoder----------------------------------
        with tf.variable_scope('dec'):
            # decode for training sent
            with tf.variable_scope('sent', initializer=tf.contrib.layers.xavier_initializer(), dtype = tf.float32, reuse=False):
                self.dec_cell = tf.contrib.rnn.GRUCell(self.config.dim_hidden)
                self.dec_cell = tf.contrib.rnn.DropoutWrapper(self.dec_cell, output_keep_prob = self.t_variables['dec_keep_prob'])
                self.dec_sent_cell =self.dec_cell
                self.latent_hidden_layer = tf.layers.Dense(units=self.config.dim_hidden, activation=tf.nn.relu, name='latent_hidden_linear')
                self.dec_sent_initial_state = self.latent_hidden_layer(self.latents_sent_posterior) # batch_l x max_doc_l x dim_hidden
                self.output_layer = tf.layers.Dense(self.config.n_vocab, use_bias=False, name='out')
                
                if self.config.attention:
                    self.sent_outputs_flat = tf.reshape(self.sent_outputs, [self.batch_l*self.max_doc_l, self.max_sent_l, self.config.dim_hidden*2])
                    self.att_sent_l_flat = tf.reshape(tf.maximum(self.sent_l, 1), [self.batch_l*self.max_doc_l])
                    self.att_sent_mechanism = tf.contrib.seq2seq.LuongAttention(num_units=self.config.dim_hidden, 
                                                                                memory=self.sent_outputs_flat, \
                                                                                memory_sequence_length=self.att_sent_l_flat)
                    self.att_cell = tf.contrib.seq2seq.AttentionWrapper(self.dec_cell, 
                                                                                attention_mechanism=self.att_sent_mechanism, 
                                                                                attention_layer_size=self.config.dim_hidden)
                    self.dec_sent_cell = self.att_cell

                # teacher forcing
                self.dec_input_idxs = self.t_variables['dec_input_idxs'] # batch_l x max_doc_l x max_dec_sent_l
                self.dec_inputs = tf.nn.embedding_lookup(self.embeddings, self.dec_input_idxs) # batch_l x max_doc_l x max_dec_sent_l x dim_emb

                # output_sent_l == dec_sent_l
                self.output_logits_flat, self.output_sent_l_flat = decode_output_logits_flat(self,
                                                    dec_cell=self.dec_sent_cell,
                                                    dec_initial_state=self.dec_sent_initial_state, 
                                                    dec_inputs=self.dec_inputs,
                                                    dec_sent_l=self.dec_sent_l,
                                                    latents_input=self.latents_sent_posterior) # batch_l*max_doc_l x max_output_sent_l x n_vocab

                self.output_sent_l = tf.reshape(self.output_sent_l_flat, [self.batch_l, self.max_doc_l])
                self.max_output_sent_l = tf.reduce_max(self.output_sent_l)
                self.output_logits = tf.reshape(self.output_logits_flat, \
                                    [self.batch_l, self.max_doc_l, self.max_output_sent_l, self.config.n_vocab], name='output_logits')
                if self.config.disc_gumbel:
                    self.output_input_idxs = sample_gumbels(self.output_logits, self.softmax_temperature, self.config.seed, self.sample) # batch_l x max_doc_l x max_output_sent_l  x n_vocab
                else:
                    self.output_input_idxs = self.output_logits
            
            # decode for training topic probs
            with tf.variable_scope('sent', initializer=tf.contrib.layers.xavier_initializer(), dtype = tf.float32, reuse=True):                
                self.dec_topic_cell = self.dec_cell
                if self.config.attention:
                    self.topic_outputs_flat = tf.contrib.seq2seq.tile_batch(tf.reshape(self.sent_outputs, \
                                            [self.batch_l, self.max_doc_l*self.max_sent_l, self.sent_outputs.get_shape()[-1]]), \
                                            multiplier=self.config.n_topic) # batch_l*n_topic x max_doc_l*max_sent_l x dim_hidden*2
                    self.score_mask = tf.contrib.seq2seq.tile_batch(tf.reshape(tf.sequence_mask(self.sent_l), \
                                            [self.batch_l, self.max_doc_l*self.max_sent_l]), multiplier=self.config.n_topic) # batch_l*n_topic x max_doc_l*max_sent_l
                    self.hier_score = tf.reshape(tf.transpose(self.probs_sent_topic_posterior, [0, 2, 1]), \
                                            [self.batch_l*self.config.n_topic, self.max_doc_l]) # batch_l*n_topic x max_doc_l
                    
                    self.att_topic_mechanism = HierarchicalAttention(num_units=self.config.dim_hidden, 
                                                            memory=self.topic_outputs_flat,
                                                            score_mask=self.score_mask,
                                                            hier_score=self.hier_score)
                    self.att_topic_cell = AttentionWrapper(self.dec_cell, 
                                                            attention_mechanism=self.att_topic_mechanism, 
                                                            attention_layer_size=self.config.dim_hidden)
                    self.dec_topic_cell = self.att_topic_cell
                
                if not self.config.disc_mean:
                    self.dec_topic_initial_state = self.latent_hidden_layer(self.latents_topic_posterior)
                    dec_topic_outputs, self.summary_sent_l_flat = decode_output_sample_flat(self, 
                                                            dec_cell=self.dec_topic_cell,
                                                            dec_initial_state=self.dec_topic_initial_state,
                                                            softmax_temperature=self.softmax_temperature,
                                                            sample=self.sample,
                                                            latents_input=self.latents_topic_posterior) # batch_l*max_doc_l x max_summary_sent_l x n_vocab
                else:
                    self.dec_topic_initial_state = self.latent_hidden_layer(self.means_topic_posterior)
                    dec_topic_outputs, self.summary_sent_l_flat = decode_output_sample_flat(self, 
                                                            dec_cell=self.dec_topic_cell,
                                                            dec_initial_state=self.dec_topic_initial_state,
                                                            softmax_temperature=self.softmax_temperature,
                                                            sample=self.sample,
                                                            latents_input=self.means_topic_posterior) # batch_l*max_doc_l x max_summary_sent_l x n_vocab
                
                self.summary_sent_l = tf.reshape(self.summary_sent_l_flat, [self.batch_l, self.config.n_topic])
                self.max_summary_sent_l = tf.reduce_max(self.summary_sent_l)
                if self.config.disc_gumbel:
                    summary_input_idxs_flat = dec_topic_outputs.sample_id
                else:
                    summary_input_idxs_flat = dec_topic_outputs.rnn_output
                self.summary_input_idxs = tf.reshape(summary_input_idxs_flat, \
                                                     [self.batch_l, self.config.n_topic, self.max_summary_sent_l, self.config.n_vocab], name='summary_input_idxs')
                
                # re-encode topic sentence outputs
                self.summary_inputs = tf.tensordot(self.summary_input_idxs, self.embeddings, axes=[[-1], [0]]) # batch_l x n_topic x max_summary_sent_l x dim_emb
                self.summary_input_sent_l = self.summary_sent_l - 1 # to remove EOS
                self.mask_summary_sent = tf.sequence_mask(self.summary_input_sent_l, \
                                                          maxlen=self.max_summary_sent_l, dtype=tf.float32) # batch_l x n_topic x max_summary_sent_l 
                self.mask_summary_doc = tf.ones([self.batch_l, self.config.n_topic], dtype=tf.float32)
                
            # beam decode for inference of original sentences
            with tf.variable_scope('sent', initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, reuse=True):
                self.beam_dec_sent_cell = self.dec_cell
                if self.config.attention:
                    self.beam_sent_outputs_flat = tf.contrib.seq2seq.tile_batch(self.sent_outputs_flat, multiplier=self.config.beam_width)
                    self.beam_att_sent_l_flat = tf.contrib.seq2seq.tile_batch(self.att_sent_l_flat, multiplier=self.config.beam_width)
                    self.beam_att_sent_mechanism = tf.contrib.seq2seq.LuongAttention(num_units=self.config.dim_hidden,
                                                                                                           memory=self.beam_sent_outputs_flat,
                                                                                                           memory_sequence_length=self.beam_att_sent_l_flat)
                    self.beam_dec_sent_cell = tf.contrib.seq2seq.AttentionWrapper(self.beam_dec_sent_cell,
                                                                                             attention_mechanism=self.beam_att_sent_mechanism, 
                                                                                             attention_layer_size=self.config.dim_hidden)

                # infer original sentences
                self.beam_output_idxs, _, _= decode_beam_output_token_idxs(self,
                                                                    beam_dec_cell=self.beam_dec_sent_cell,
                                                                    dec_initial_state=self.dec_sent_initial_state,
                                                                    latents_input=self.means_sent_posterior,
                                                                    name='beam_output_idxs')
            
            # beam decode for inference of topic sentences
            with tf.variable_scope('sent', initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, reuse=True):
                self.beam_dec_topic_cell = self.dec_cell
                if self.config.attention:
                    self.beam_topic_outputs_flat = tf.contrib.seq2seq.tile_batch(self.topic_outputs_flat, multiplier=self.config.beam_width)
                    self.beam_score_mask = tf.contrib.seq2seq.tile_batch(self.score_mask, multiplier=self.config.beam_width)
                    self.beam_hier_score = tf.contrib.seq2seq.tile_batch(self.hier_score, multiplier=self.config.beam_width)
                    self.beam_att_topic_mechanism = HierarchicalAttention(num_units=self.config.dim_hidden, 
                                                                                        memory=self.beam_topic_outputs_flat,
                                                                                        score_mask=self.beam_score_mask,
                                                                                        hier_score=self.beam_hier_score)
                    self.beam_dec_topic_cell = AttentionWrapper(self.beam_dec_topic_cell,
                                                                                         attention_mechanism=self.beam_att_topic_mechanism, 
                                                                                         attention_layer_size=self.config.dim_hidden)
                
                # infer topic sentences
                self.beam_summary_idxs, _, _ = decode_beam_output_token_idxs(self,
                                                                    beam_dec_cell=self.beam_dec_topic_cell,
                                                                    dec_initial_state=self.dec_topic_initial_state,
                                                                    latents_input=self.latents_topic_posterior,
                                                                    name='beam_summary_idxs')
                
                self.beam_mask_summary_sent = tf.logical_not(tf.equal(self.beam_summary_idxs, \
                                                                      self.config.EOS_IDX)) # batch_l x n_topic x max_summary_sent_l
                self.beam_summary_input_sent_l = tf.reduce_sum(tf.cast(self.beam_mask_summary_sent, tf.int32), -1) # batch_l x n_topic
                beam_summary_soft_idxs = tf.one_hot(tf.where(self.beam_mask_summary_sent, \
                                                                            self.beam_summary_idxs, tf.zeros_like(self.beam_summary_idxs)), depth=self.config.n_vocab)
                self.beam_summary_inputs = tf.tensordot(beam_summary_soft_idxs, \
                                                        self.embeddings, [[-1], [0]]) # batch_l x n_topic x max_beam_summary_sent_l x dim_emb

        # ------------------------------Discriminator------------------------------                
        # encode by MLP
        if self.config.enc == 'mlp':
            with tf.variable_scope('disc'):
                with tf.variable_scope('prob_topic', reuse=True):
                    self.summary_state = encode_states(self, enc_inputs=self.summary_inputs, mask_sent=self.mask_summary_sent, \
                                                                   enc_keep_prob=self.enc_keep_prob, config=self.config) # batch_l x n_topic x dim_hidden
        elif self.config.enc == 'bow':
            with tf.variable_scope('disc'):
                with tf.variable_scope('prob_topic', reuse=True):
                    self.bow_summary_input_idxs = tf.multiply(self.summary_input_idxs, self.mask_bow)
                    self.bow_summary_inputs = tf.tensordot(self.bow_summary_input_idxs, self.embeddings, axes=[[-1], [0]]) # batch_l x max_doc_l x max_sent_l x dim_emb
                    self.mask_summary_bow = tf.reduce_sum(self.bow_summary_input_idxs, -1)
                    self.summary_state = encode_states(self, enc_inputs=self.bow_summary_inputs, mask_sent=self.mask_summary_bow, \
                                                                   enc_keep_prob=self.enc_keep_prob, config=self.config) # batch_l x max_doc_l x dim_hidden
        elif self.config.enc == 'rnn':
            with tf.variable_scope('emb'):
                with tf.variable_scope('sent', reuse=True):
                    _, self.summary_state = encode_inputs(self, enc_inputs=self.summary_inputs, sent_l=self.summary_input_sent_l) # batch_l x max_doc_l x dim_hidden*2
                    _, self.beam_summary_state = encode_inputs(self, enc_inputs=self.beam_summary_inputs, sent_l=self.beam_summary_input_sent_l) # batch_l x max_doc_l x dim_hidden*2

        with tf.variable_scope('disc'):
            with tf.variable_scope('prob_topic', reuse=True):
                self.probs_summary_topic_posterior, _, _ = \
                        encode_gsm_probs_topic_posterior(self, self.summary_state.get_shape()[-1], self.summary_state, self.mask_summary_doc, self.config)
                self.logits_summary_topic_posterior_ = tf_log(tf.matrix_diag_part(self.probs_summary_topic_posterior)) # batch_l x n_topic
                self.logits_summary_topic_posterior = tf_clip_vals(self.logits_summary_topic_posterior_, self.probs_topic_posterior)
                             
        # ------------------------------Optimizer and Loss------------------------------            
        with tf.name_scope('opt'):                    
            partition_doc = tf.cast(self.mask_doc, dtype=tf.int32)
            self.n_sents = tf.cast(tf.reduce_sum(self.doc_l), dtype=tf.float32)
            self.n_tokens = tf.reduce_sum(self.dec_sent_l)
            
            # ------------------------------Reconstruction Loss of Language Model------------------------------            
            # target and mask
            self.dec_target_idxs = self.t_variables['dec_target_idxs'] # batch_l x max_doc_l x max_dec_sent_l
            self.dec_sent_l = self.t_variables['dec_sent_l'] # batch_l x max_doc_l
            self.max_dec_sent_l = tf.reduce_max(self.dec_sent_l) # = max_sent_l + 1
            self.dec_mask_sent = tf.sequence_mask(self.dec_sent_l, maxlen=self.max_dec_sent_l, dtype=tf.float32)
            self.dec_target_idxs_flat = tf.reshape(self.dec_target_idxs, [self.batch_l*self.max_doc_l, self.max_dec_sent_l])
            self.dec_mask_sent_flat = tf.reshape(self.dec_mask_sent, [self.batch_l*self.max_doc_l, self.max_dec_sent_l])

            # nll for each token (summed over sentence)
            self.recon_max_sent_l = tf.minimum(self.max_dec_sent_l, self.max_output_sent_l) if self.config.sample else None
            losses_recon_flat = tf.reduce_sum(tf.contrib.seq2seq.sequence_loss(self.output_logits_flat[:, :self.recon_max_sent_l, :], 
                                                                                      self.dec_target_idxs_flat[:, :self.recon_max_sent_l], 
                                                                                      self.dec_mask_sent_flat[:, :self.recon_max_sent_l], 
                                                                                      average_across_timesteps=False,
                                                                                      average_across_batch=False), -1) # batch_l*max_doc_l
            self.losses_recon = tf.reshape(losses_recon_flat, [self.batch_l, self.max_doc_l])
            self.loss_recon = tf.reduce_mean(tf.dynamic_partition(self.losses_recon, partition_doc, num_partitions=2)[1]) # average over doc x batch

            # ------------------------------KL divergence Loss of Topic Probability Distribution------------------------------
            if self.config.topic_model:
                self.probs_sent_topic_prior = tf.expand_dims(self.probs_doc_topic_posterior, 1) # batch_l x 1 x n_topic
            else:
                self.probs_sent_topic_prior = tf.ones_like(self.probs_sent_topic_posterior, dtype=tf.float32) / \
                                                        self.config.n_topic # batch_l x max_doc_l x n_topic, uniform distribution over topics
            self.losses_kl_prob = tf.reduce_sum(tf.multiply(self.probs_sent_topic_posterior, \
                                                            (tf_log(self.probs_sent_topic_posterior)-tf_log(self.probs_sent_topic_prior))), -1)
            self.loss_kl_prob = tf.reduce_mean(tf.dynamic_partition(self.losses_kl_prob, partition_doc, num_partitions=2)[1]) # average over doc x batch
            
            # ------------------------------KL divergence Loss of Sentence Latents Distribution------------------------------
            self.losses_kl_sent_gauss = compute_kl_losses_sent_gauss(self) # batch_l x max_doc_l x n_topic, sum over latent dimension
            self.losses_kl_sent_gmm = tf.reduce_sum(tf.multiply(self.probs_sent_topic_posterior, self.losses_kl_sent_gauss), -1) # batch_l x max_doc_l, sum over topics
            self.loss_kl_sent_gmm = tf.reduce_mean(tf.dynamic_partition(self.losses_kl_sent_gmm, partition_doc, num_partitions=2)[1]) # average over doc x batch

            # ------------------------------KL divergence Loss of Topic Latents Distribution------------------------------
            if self.config.reverse_kl:
                self.losses_kl_topic_pairs_gauss = compute_kl_losses_topic_paris_gauss(self)
                self.losses_kl_topic_gauss_reverse = tf.reduce_sum(self.losses_kl_topic_pairs_gauss * self.config.mask_tree[None, None, :, :], -1) / \
                                        np.maximum(np.sum(self.config.mask_tree[None, None, :, :], -1), 1) # batch_l x 1 x n_topic, mean over other child topics
                self.losses_kl_topic_gmm_reverse = tf.reduce_sum(tf.multiply(self.probs_sent_topic_posterior, self.losses_kl_topic_gauss_reverse), -1) # batch_l x max_doc_l, sum over topics
                self.loss_kl_topic_gmm_reverse = tf.reduce_mean(tf.dynamic_partition(self.losses_kl_topic_gmm_reverse, partition_doc, num_partitions=2)[1])
            else:
                self.loss_kl_topic_gmm_reverse = tf.constant(0., dtype=tf.float32)
                
            # for monitor
            self.losses_kl_topic_gauss = compute_kl_losses_topic_gauss(self) # batch_l x 1 x n_topic, sum over latent dimension
            self.losses_kl_topic_gmm = tf.reduce_sum(tf.multiply(self.probs_sent_topic_posterior, self.losses_kl_topic_gauss), -1) # batch_l x max_doc_l, sum over topics
            self.loss_kl_topic_gmm = tf.reduce_mean(tf.dynamic_partition(self.losses_kl_topic_gmm, partition_doc, num_partitions=2)[1])

            # ------------------------------KL divergence Loss of Root State Distribution------------------------------
            if self.config.prior_root:
                self.losses_kl_root = compute_kl_losses(self.means_state_root_posterior, self.logvars_state_root_posterior) # batch_l x max_doc_l
                self.loss_kl_root = tf.reduce_sum(self.losses_kl_root) / tf.cast(tf.reduce_sum(self.doc_l), dtype=tf.float32) # average over doc x batch
            else:
                self.loss_kl_root = tf.constant(0, dtype=tf.float32)
            
            # ------------------------------Discriminator Loss------------------------------
            if self.config.disc_topic:
                self.losses_disc_topic = -tf.reduce_sum(self.logits_summary_topic_posterior, -1) # batch_l, sum over topic
                self.loss_disc_topic = tf.reduce_sum(self.losses_disc_topic) / self.n_sents # average over doc x batch
            else:
                self.loss_disc_topic = tf.constant(0, dtype=tf.float32)
                                
            # ------------------------------Loss of Topic Model------------------------------
            if self.config.topic_model:
                # recon
                self.topic_losses_recon = -tf.reduce_sum(tf.multiply(self.t_variables['doc_bows'], self.logits_bow), -1) # n_batch, sum over n_bow
                self.topic_loss_recon = tf.reduce_mean(self.topic_losses_recon) # average over doc x batch

                # kl_bow
                self.means_topic_bow_prior = tf.squeeze(get_params_topic_prior(self, tf.expand_dims(self.means_topic_bow_posterior, 0), \
                                                                    tf.zeros([1, self.config.dim_latent], dtype=tf.float32)), 0) # n_topic x dim_latent
                self.logvars_topic_bow_prior = tf.squeeze(get_params_topic_prior(self, tf.expand_dims(self.logvars_topic_bow_posterior, 0), \
                                                                                tf.zeros([1, self.config.dim_latent], dtype=tf.float32)), 0) # n_topic x dim_latent
                self.topic_losses_kl_bow = compute_kl_losses(self.means_topic_bow_posterior, self.logvars_topic_bow_posterior, \
                                                                            means_prior=self.means_topic_bow_prior, logvars_prior=self.logvars_topic_bow_prior) # n_topic
                self.topic_loss_kl_bow = tf.reduce_mean(self.topic_losses_kl_bow) # average over doc x batch
                
                # kl_prob
                self.topic_losses_kl_prob = compute_kl_losses(self.means_probs_doc_topic_posterior, self.logvars_probs_doc_topic_posterior) # batch_l
                self.topic_loss_kl_prob = tf.reduce_mean(self.topic_losses_kl_prob) # average over doc x batch
            else:
                self.topic_loss_recon = tf.constant(0, dtype=tf.float32)
                self.topic_loss_kl_bow = tf.constant(0, dtype=tf.float32)
                self.topic_loss_kl_prob = tf.constant(0, dtype=tf.float32)
                
            # ------------------------------Topic Regularization Loss------------------------------
            if self.config.reg != '':
                if self.config.reg == 'mean':
                    self.topic_dots = self.get_topic_dots(self.means_topic_posterior) # batch_l x n_topic-1 x n_topic-1
                elif self.config.reg == 'bow':
                    self.topic_dots = self.get_topic_dots(tf.expand_dims(self.topic_bow, 0)) # batch_l(=1) x n_topic-1 x n_topic-1
                    
                self.losses_reg = tf.reduce_sum(tf.square(self.topic_dots - tf.eye(len(self.config.all_child_idxs))) * self.config.mask_tree_reg, [1, 2])\
                                        / tf.reduce_sum(self.config.mask_tree_reg) # batch_l
                self.loss_reg = tf.reduce_mean(self.losses_reg) # average over batch
            else:
                self.loss_reg = tf.constant(0, dtype=tf.float32)
            
            # ------------------------------Optimizer------------------------------
            if self.config.anneal == 'linear':
                self.tau = tf.cast(tf.divide(self.global_step, tf.constant(self.config.linear_steps)), dtype=tf.float32)
                self.beta = tf.minimum(1., self.config.beta_init + self.tau)
            elif self.config.anneal == 'cycle':
                self.tau = tf.cast(tf.divide(tf.mod(self.global_step, tf.constant(self.config.cycle_steps)), tf.constant(self.config.cycle_steps)), dtype=tf.float32)
                self.beta = tf.minimum(1., self.config.beta_init + self.tau/(1.-self.config.r_cycle))
            else:
                self.beta = tf.constant(1.)
                
            self.beta_disc = self.beta if self.config.beta_disc else tf.constant(1.)
            
            def get_opt(loss, var_list, lr, global_step=None):
                if self.config.opt == 'adam':
                    Optimizer = tf.train.AdamOptimizer
                elif self.config.opt == 'adagrad':
                    Optimizer = tf.train.AdagradOptimizer
                
                optimizer = Optimizer(lr)
                grad_vars = optimizer.compute_gradients(loss=loss, var_list=var_list)
                clipped_grad_vars = [(tf.clip_by_value(grad, -self.config.grad_clip, self.config.grad_clip), var) for grad, var in grad_vars if grad is not None]
                opt = optimizer.apply_gradients(clipped_grad_vars, global_step=global_step)
                return opt, grad_vars, clipped_grad_vars
                
            # ------------------------------Loss Setting------------------------------
            if self.config.turn:
                self.loss = self.loss_recon + \
                             self.beta * tf.maximum(tf.maximum(self.loss_kl_sent_gmm, self.config.capacity_gmm) \
                                                            - self.loss_kl_topic_gmm_reverse, self.config.margin_gmm) + \
                             self.beta * self.loss_kl_root + \
                             self.topic_loss_recon + \
                             self.beta * self.topic_loss_kl_bow + \
                             self.beta * self.topic_loss_kl_prob + \
                             self.config.lam_reg * self.loss_reg

                self.opt, self.grad_vars, self.clipped_grad_vars = \
                    get_opt(self.loss, var_list=list(tf.trainable_variables('emb') + tf.trainable_variables('enc') + tf.trainable_variables('dec')), \
                                lr=self.config.lr, global_step=self.global_step)

                self.loss_disc = self.beta_disc * self.config.lam_disc * self.loss_disc_topic + \
                                    self.beta * tf.maximum(self.loss_kl_prob, self.config.capacity_prob)

                self.opt_disc, self.grad_vars_disc, self.clipped_grad_vars_disc = \
                    get_opt(self.loss_disc, var_list=list(tf.trainable_variables('emb') + tf.trainable_variables('disc')), lr=self.config.lr_disc)

            else:
                self.loss = self.loss_recon + \
                             self.beta * tf.maximum(tf.maximum(self.loss_kl_sent_gmm, self.config.capacity_gmm) \
                                                            - self.loss_kl_topic_gmm_reverse, self.config.margin_gmm) + \
                             self.beta * self.loss_kl_root + \
                             self.topic_loss_recon + \
                             self.beta * self.topic_loss_kl_bow + \
                             self.beta * self.topic_loss_kl_prob + \
                             self.beta_disc * self.config.lam_disc * self.loss_disc_topic + \
                             self.beta * tf.maximum(self.loss_kl_prob, self.config.capacity_prob) + \
                             self.config.lam_reg * self.loss_reg
                self.loss_disc = tf.constant(0, dtype=tf.float32)

                self.opt, self.grad_vars, self.clipped_grad_vars = \
                    get_opt(self.loss, var_list=tf.trainable_variables(), lr=self.config.lr, global_step=self.global_step)
                self.opt_disc = tf.constant(0, dtype=tf.float32)

            
            # ------------------------------Evaluatiion------------------------------
            self.loss_list_train = [self.loss, self.loss_disc, self.loss_recon, self.loss_kl_prob, self.loss_kl_sent_gmm, self.loss_kl_topic_gmm_reverse, \
                self.loss_kl_root, self.loss_disc_topic, self.topic_loss_recon, self.topic_loss_kl_bow, self.topic_loss_kl_prob, self.loss_reg, tf.constant(0)]
            self.loss_list_eval = [self.loss, self.loss_disc, self.loss_recon, self.loss_kl_prob, self.loss_kl_sent_gmm, self.loss_kl_topic_gmm_reverse, \
                self.loss_kl_root, self.loss_disc_topic, self.topic_loss_recon, self.topic_loss_kl_bow, self.topic_loss_kl_prob, self.loss_reg, self.loss_kl_topic_gmm]
            self.loss_sum = (self.loss_recon + self.loss_kl_prob + self.loss_kl_sent_gmm + self.loss_kl_root + self.loss_disc_topic + \
                                 self.topic_loss_recon + self.topic_loss_kl_bow + self.topic_loss_kl_prob) * self.n_sents
                        
    def get_topic_dots(self, means):
        diff_means = tf.concat([tf.expand_dims(means[:, self.config.topic_idxs.index(child_idx), :] - \
                            means[:, self.config.topic_idxs.index(self.config.child_to_parent_idxs[child_idx]), :], 1) for child_idx in self.config.all_child_idxs], axis=1)
        diff_means_norm = diff_means / tf.norm(diff_means, axis=-1, keepdims=True) # batch_l x n_topic-1 x dim_latent
        topic_dots = tf.clip_by_value(tf.matmul(diff_means_norm, tf.transpose(diff_means_norm, [0, 2, 1])), -1., 1.) # batch_l x n_topic-1 x n_topic-1
        return topic_dots            
            
    def get_feed_dict(self, batch, mode, assertion=False):
        batch_l = len(batch)

        doc_l = np.array([instance.doc_l for instance in batch])
        max_doc_l = np.max(doc_l)

        sent_l = np.array([[instance.sent_l[i] if i < instance.doc_l else 0 for i in range(max_doc_l)] for instance in batch])
        max_sent_l = np.max(sent_l)
        
        dec_sent_l = np.array([[instance.sent_l[i] + 1 if i < instance.doc_l else 0 for i in range(max_doc_l)] for instance in batch])
        
        bows_list = [instance.bows.toarray().astype(np.float32) for instance in batch]
        bows = np.array([[bows[i] if i < instance.doc_l else np.zeros(self.config.dim_bow, dtype=np.float32) for i in range(max_doc_l)] for instance, bows in zip(batch, bows_list)])
        doc_bows = np.array([instance.doc_bow for instance in batch])
        
        pad_token_idxs = lambda token_idxs_list, max_sent_l: np.array([pad_sequences(doc_token_idxs, maxlen=max_sent_l, padding='post', value=self.config.PAD_IDX, dtype=np.int32) for doc_token_idxs in token_idxs_list])
        def token_dropout(sent_idxs, mode):
            sent_idxs_dropout = np.asarray(sent_idxs)
            if mode == 'train':
                word_keep_prob = self.config.word_keep_prob
            elif mode == 'eval':
                word_keep_prob = 1.
            sent_idxs_dropout[np.random.rand(len(sent_idxs)) > word_keep_prob] = self.config.UNK_IDX
            return list(sent_idxs_dropout)
        
        enc_input_idxs_list = [[instance.token_idxs[i] if i < instance.doc_l else [] for i in range(max_doc_l)] for instance in batch]
        dec_input_idxs_list = [[[self.config.BOS_IDX] + token_dropout(sent_idxs, mode) for sent_idxs in doc_idxs] for doc_idxs in enc_input_idxs_list]
        dec_target_idxs_list = [[sent_idxs + [self.config.EOS_IDX] for sent_idxs in doc_idxs] for doc_idxs in enc_input_idxs_list]

        enc_input_idxs = pad_token_idxs(enc_input_idxs_list, max_sent_l)
        dec_input_idxs = pad_token_idxs(dec_input_idxs_list, max_sent_l+1)
        dec_target_idxs = pad_token_idxs(dec_target_idxs_list, max_sent_l+1)

        assert len(bows) == len(enc_input_idxs)
        assert (batch_l, max_doc_l, self.config.dim_bow) == (bows.shape)
        assert (batch_l, max_doc_l, max_sent_l) == enc_input_idxs.shape
        assert (batch_l, max_doc_l, max_sent_l+1) == dec_input_idxs.shape == dec_target_idxs.shape

        if mode == 'train':
            sample = True
            train = True
            enc_keep_prob = self.config.enc_keep_prob
            dec_keep_prob = self.config.dec_keep_prob
        elif mode == 'eval':
            sample = False
            train = False
            enc_keep_prob = dec_keep_prob = 1.

        feed_dict = {
                    self.t_variables['bows']: bows, self.t_variables['doc_bows']: doc_bows, 
                    self.t_variables['batch_l']: batch_l, self.t_variables['doc_l']: doc_l, self.t_variables['sent_l']: sent_l, self.t_variables['dec_sent_l']: dec_sent_l, 
                    self.t_variables['enc_input_idxs']: enc_input_idxs, self.t_variables['dec_input_idxs']: dec_input_idxs, self.t_variables['dec_target_idxs']: dec_target_idxs, 
                    self.t_variables['enc_keep_prob']: enc_keep_prob, self.t_variables['dec_keep_prob']: dec_keep_prob, 
                    self.t_variables['sample']: sample
        }
        return  feed_dict