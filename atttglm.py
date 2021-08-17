#coding: utf-8
import pdb
import random
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

from components import tf_log, tf_clip_vals, tf_clip_means, tf_clip_covs, sample_latents_fullcov, sample_gumbels, get_params_topic_prior, \
                                            compute_kl_losses, compute_kl_losses_sent_gauss, compute_kl_losses_topic_gauss, compute_losses_coverage
from seq2seq import get_embeddings, encode_inputs, encode_latents_gauss, encode_nhdp_probs_topic_posterior, decode_output_logits_flat, decode_output_sample_flat, decode_beam_output_token_idxs, decode_sample_output_token_idxs, wrap_attention

class AttentionTopicGuidedLanguageModel():
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
        t_variables['n_oov'] = tf.placeholder(tf.int32, name='n_oov')
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

            self.enc_keep_prob = self.t_variables['enc_keep_prob']
        
        # ------------------------------Encoder------------------------------        
        with tf.variable_scope('enc'):
            with tf.variable_scope('word', reuse=False):
                self.enc_embeddings = get_embeddings(self)
                # get sentence embeddings
                self.enc_inputs = tf.nn.embedding_lookup(self.enc_embeddings, self.t_variables['enc_input_idxs']) # batch_l x max_doc_l x max_sent_l x dim_emb
            
            with tf.variable_scope('sent', reuse=False):
                self.sent_outputs, self.sent_state = \
                    encode_inputs(self, enc_inputs=self.enc_inputs, sent_l=self.sent_l) # batch_l x max_doc_l x dim_hidden*2
                self.memory_idxs = self.t_variables['enc_input_idxs']
                                    
            # get sentence latents
            with tf.variable_scope('latents_sent', reuse=False):
                self.latents_sent_posterior, self.means_sent_posterior, self.logvars_sent_posterior = \
                        encode_latents_gauss(self.sent_state, dim_latent=self.config.dim_latent, sample=self.sample, \
                                             config=self.config, name='sent_posterior', min_logvar=self.config.min_logvar) # batch_l x max_doc_l x dim_latent
                
        # ------------------------------Discriminator (Topic Encoder)------------------------------        
        with tf.variable_scope('disc'):            
            with tf.variable_scope('prob_topic', reuse=False):   
                self.probs_sent_topic_posterior, self.tree_sent_sticks_path, self.tree_sent_sticks_depth = \
                    encode_nhdp_probs_topic_posterior(self, self.sent_state.get_shape()[-1], self.sent_state, self.mask_doc, self.config) # batch_l x max_doc_l x n_topic
                        
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

                self.mean_root_prior = tf.zeros([self.batch_l, self.config.dim_latent], dtype=tf.float32)
                self.cov_root_prior = tf.eye(self.config.dim_latent, batch_shape=[self.batch_l], dtype=tf.float32) * self.config.cov_root
                self.means_topic_prior = get_params_topic_prior(self, self.means_topic_posterior, self.mean_root_prior) # batch_l x n_topic x dim_latent
                self.covs_topic_prior = get_params_topic_prior(self, self.covs_topic_posterior, self.cov_root_prior) # batch_l x n_topic x dim_latent x dim_latent 
                
        # ------------------------------Decoder----------------------------------
        with tf.variable_scope('dec'):
            with tf.variable_scope('word', reuse=False):
                self.dec_embeddings = get_embeddings(self)
            
            # decode for training sent
            with tf.variable_scope('sent', initializer=tf.contrib.layers.xavier_initializer(), dtype = tf.float32, reuse=False):
                self.dec_cell = tf.contrib.rnn.GRUCell(self.config.dim_hidden)
                self.dec_cell = tf.contrib.rnn.DropoutWrapper(self.dec_cell, output_keep_prob = self.t_variables['dec_keep_prob'])
                
                self.latent_hidden_layer = tf.layers.Dense(units=self.config.dim_hidden, activation=tf.nn.relu, name='latent_hidden_linear')
                self.dec_sent_initial_state = self.latent_hidden_layer(self.latents_sent_posterior) # batch_l x max_doc_l x dim_hidden
                self.output_layer = tf.layers.Dense(self.config.n_vocab, use_bias=False, dtype=tf.float32, name='output')
                self.prob_gen_layer = None
                self.pointer_layer = None
                    
                if self.config.attention:
                    self.dec_sent_cell = wrap_attention(self, self.dec_cell, self.sent_outputs, n_tiled=self.max_doc_l)
                else:
                    self.dec_sent_cell = self.dec_cell
                    
                # teacher forcing
                self.dec_input_idxs = self.t_variables['dec_input_idxs'] # batch_l x max_doc_l x max_dec_sent_l
                self.dec_inputs = tf.nn.embedding_lookup(self.dec_embeddings, self.dec_input_idxs) # batch_l x max_doc_l x max_dec_sent_l x dim_emb

                # output_sent_l == dec_sent_l
                self.dec_sent_outputs, self.dec_sent_final_state, self.output_sent_l_flat = decode_output_logits_flat(self,
                                                    dec_cell=self.dec_sent_cell,
                                                    dec_initial_state=self.dec_sent_initial_state, 
                                                    dec_inputs=self.dec_inputs,
                                                    dec_sent_l=self.dec_sent_l,
                                                    latents_input=self.latents_sent_posterior) # batch_l*max_doc_l x max_output_sent_l x n_vocab

                self.output_logits_flat = self.dec_sent_outputs.rnn_output
                self.output_sent_l = tf.reshape(self.output_sent_l_flat, [self.batch_l, self.max_doc_l])
                self.max_output_sent_l = tf.reduce_max(self.output_sent_l)
                self.output_logits = tf.reshape(self.output_logits_flat, \
                                                [self.batch_l, self.max_doc_l, self.max_output_sent_l, tf.shape(self.output_logits_flat)[-1]], name='output_logits')
                self.output_input_idxs = sample_gumbels(self.output_logits, self.softmax_temperature, self.config.seed, self.sample) # batch_l x max_doc_l x max_output_sent_l  x n_vocab
                    
                # re-encode original sentence outputs
                self.output_inputs = tf.tensordot(self.output_input_idxs, self.enc_embeddings, axes=[[-1], [0]]) # batch_l x max_doc_l x max_sent_l x dim_emb
                self.output_input_sent_l = self.output_sent_l - 1 # to remove EOS
                self.mask_output_doc = self.mask_doc
            
            # decode for training topic probs
            with tf.variable_scope('sent', initializer=tf.contrib.layers.xavier_initializer(), dtype = tf.float32, reuse=True):                
                if self.config.attention: 
                    self.dec_topic_cell = wrap_attention(self, self.dec_cell, self.sent_outputs, n_tiled=self.config.n_topic)
                else:
                    self.dec_topic_cell = self.dec_cell
                
                self.dec_topic_initial_state = self.latent_hidden_layer(self.means_topic_posterior)
                self.dec_topic_outputs, self.dec_topic_final_state, self.summary_sent_l_flat = decode_output_sample_flat(self, 
                                                        dec_cell=self.dec_topic_cell,
                                                        dec_initial_state=self.dec_topic_initial_state,
                                                        softmax_temperature=self.softmax_temperature,
                                                        sample=self.sample,
                                                        latents_input=self.means_topic_posterior) # batch_l*n_topic x max_summary_sent_l x n_vocab
            
                self.summary_sent_l = tf.reshape(self.summary_sent_l_flat, [self.batch_l, self.config.n_topic])
                self.max_summary_sent_l = tf.reduce_max(self.summary_sent_l)
                
                summary_input_idxs_flat = self.dec_topic_outputs.sample_id
                self.summary_input_idxs = tf.reshape(summary_input_idxs_flat, \
                                                                     [self.batch_l, self.config.n_topic, self.max_summary_sent_l, tf.shape(summary_input_idxs_flat)[-1]], name='summary_input_idxs')
                
                # re-encode topic sentence outputs
                self.summary_inputs = tf.tensordot(self.summary_input_idxs, self.enc_embeddings, axes=[[-1], [0]]) # batch_l x n_topic x max_summary_sent_l x dim_emb
                self.summary_input_sent_l = self.summary_sent_l - 1 # to remove EOS
                self.mask_summary_doc = tf.ones([self.batch_l, self.config.n_topic], dtype=tf.float32)
                
            # beam decode for inference of original sentences
            with tf.variable_scope('sent', initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, reuse=True):
                if self.config.nucleus < 1:
                    self.beam_output_idxs = decode_sample_output_token_idxs(self, 
                                                                            dec_cell=self.dec_sent_cell,
                                                                            dec_initial_state=self.dec_sent_initial_state,
                                                                            latents_input=self.means_sent_posterior,
                                                                            name='beam_output_idxs')
                else:
                    if self.config.attention:
                        self.beam_dec_sent_cell = wrap_attention(self, self.dec_cell, self.sent_outputs, n_tiled=self.max_doc_l, beam_width=self.config.beam_width)
                    else:
                        self.beam_dec_sent_cell = self.dec_cell

                    # infer original sentences
                    self.beam_output_idxs = decode_beam_output_token_idxs(self,
                                                                            beam_dec_cell=self.beam_dec_sent_cell,
                                                                            dec_initial_state=self.dec_sent_initial_state,
                                                                            latents_input=self.means_sent_posterior,
                                                                            name='beam_output_idxs')
            
            # beam decode for inference of topic sentences
            with tf.variable_scope('sent', initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, reuse=True):
                if self.config.nucleus < 1:
                    self.beam_summary_idxs = decode_sample_output_token_idxs(self, 
                                                                            dec_cell=self.dec_topic_cell,
                                                                            dec_initial_state=self.dec_topic_initial_state,
                                                                            latents_input=self.means_topic_posterior,
                                                                            name='beam_summary_idxs')
                else:                
                    if self.config.attention: 
                        self.beam_dec_topic_cell = wrap_attention(self, self.dec_cell, self.sent_outputs, n_tiled=self.config.n_topic, beam_width=self.config.beam_width)
                    else:
                        self.beam_dec_topic_cell = self.dec_cell

                    # infer topic sentences
                    self.beam_summary_idxs = decode_beam_output_token_idxs(self,
                                                                        beam_dec_cell=self.beam_dec_topic_cell,
                                                                        dec_initial_state=self.dec_topic_initial_state,
                                                                        latents_input=self.means_topic_posterior,
                                                                        name='beam_summary_idxs')
                
        # ------------------------------Discriminator------------------------------                
        with tf.variable_scope('enc'):
            with tf.variable_scope('sent', reuse=True):
                _, self.output_state = encode_inputs(self, enc_inputs=self.output_inputs, sent_l=self.output_input_sent_l) # batch_l x max_doc_l x dim_hidden*2                

            with tf.variable_scope('sent', reuse=True):
                _, self.summary_state = encode_inputs(self, enc_inputs=self.summary_inputs, sent_l=self.summary_input_sent_l) # batch_l x max_doc_l x dim_hidden*2
                
        with tf.variable_scope('disc'):
            with tf.variable_scope('prob_topic', reuse=True):                
                self.probs_output_topic_posterior, _, _ = \
                                            encode_nhdp_probs_topic_posterior(self, self.output_state.get_shape()[-1], self.output_state, self.mask_output_doc, self.config)
                self.logits_output_topic_posterior = tf_log(self.probs_output_topic_posterior) # batch_l x max_doc_l x n_topic
                
            with tf.variable_scope('prob_topic', reuse=True):
                self.probs_summary_topic_posterior, _, _ = \
                        encode_nhdp_probs_topic_posterior(self, self.summary_state.get_shape()[-1], self.summary_state, self.mask_summary_doc, self.config)
                self.logits_summary_topic_posterior = tf_log(tf.matrix_diag_part(self.probs_summary_topic_posterior))

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
            self.probs_sent_topic_prior = tf.tile(tf.expand_dims(tf.expand_dims(tf.constant(self.config.probs_topic_prior, dtype=tf.float32), 0), 0), [self.batch_l, self.max_doc_l, 1])
            self.losses_kl_prob = tf.reduce_sum(tf.multiply(self.probs_sent_topic_posterior, \
                                                            (tf_log(self.probs_sent_topic_posterior)-tf_log(self.probs_sent_topic_prior))), -1)
            self.loss_kl_prob = tf.reduce_mean(tf.dynamic_partition(self.losses_kl_prob, partition_doc, num_partitions=2)[1]) # average over doc x batch
            
            # ------------------------------KL divergence Loss of Sentence Latents Distribution------------------------------
            self.losses_kl_sent_gauss = compute_kl_losses_sent_gauss(self) # batch_l x max_doc_l x n_topic, sum over latent dimension
            self.losses_kl_sent_gmm = tf.reduce_sum(tf.multiply(self.probs_sent_topic_posterior, self.losses_kl_sent_gauss), -1) # batch_l x max_doc_l, sum over topics
            self.loss_kl_sent_gmm = tf.reduce_mean(tf.dynamic_partition(self.losses_kl_sent_gmm, partition_doc, num_partitions=2)[1]) # average over doc x batch

            # for monitor
            self.losses_kl_topic_gauss = compute_kl_losses_topic_gauss(self) # batch_l x 1 x n_topic, sum over latent dimension
            self.losses_kl_topic_gmm = tf.reduce_sum(tf.multiply(self.probs_sent_topic_posterior, self.losses_kl_topic_gauss), -1) # batch_l x max_doc_l, sum over topics
            self.loss_kl_topic_gmm = tf.reduce_mean(tf.dynamic_partition(self.losses_kl_topic_gmm, partition_doc, num_partitions=2)[1])
            
            # ------------------------------Discriminator Loss of Original sentences----------------------------
            self.losses_disc_sent = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.probs_sent_topic_posterior, \
                                                                    logits=self.logits_output_topic_posterior, dim=-1) # batch_l x max_doc_l, sum over topic)
            self.loss_disc_sent = tf.reduce_mean(tf.dynamic_partition(self.losses_disc_sent, partition_doc, num_partitions=2)[1])
                
            # ------------------------------Discriminator Loss of Topic sentences------------------------------
            
            self.losses_disc_topic = -tf_clip_vals(self.logits_summary_topic_posterior, self.probs_topic_posterior) # batch_l x n_topic
            self.loss_disc_topic = tf.reduce_sum(self.losses_disc_topic) / self.n_sents # average over doc x batch
                
            # ------------------------------Optimizer------------------------------
            if self.config.anneal == 'linear':
                self.tau = tf.cast(tf.divide(self.global_step, tf.constant(self.config.linear_steps)), dtype=tf.float32)
                self.beta = tf.minimum(self.config.beta_last, self.config.beta_init + self.tau)
            else:
                self.beta = tf.constant(1.)
            
            if self.config.lr_step:
                self.lr_step = tf.minimum(tf.cast(self.global_step, tf.float32)**(-1/2), tf.cast(self.global_step, tf.float32)*self.config.warmup**(-3/2))
            else:
                self.lr_step = 1.
            self.lr = self.config.lr*self.lr_step
            self.lr_disc = self.config.lr_disc*self.lr_step
            
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
                                    self.beta * tf.maximum(self.loss_kl_sent_gmm, self.config.capacity_gmm)

                self.opt, self.grad_vars, self.clipped_grad_vars = \
                    get_opt(self.loss, var_list=list(tf.trainable_variables('enc') + tf.trainable_variables('dec')), lr=self.lr, global_step=self.global_step)
                self.opt_infer, self.grad_vars_infer, self.clipped_grad_vars_infer = \
                    get_opt(self.loss, var_list=list(tf.trainable_variables('enc')), lr=self.lr, global_step=self.global_step)
                self.opt_gen, self.grad_vars_gen, self.clipped_grad_vars_gen = \
                    get_opt(self.loss, var_list=list(tf.trainable_variables('dec')), lr=self.lr, global_step=self.global_step)
                
                self.loss_disc = self.config.lam_disc * self.loss_disc_sent + \
                                            self.config.lam_disc * self.loss_disc_topic + \
                                            self.beta * tf.maximum(self.loss_kl_prob, self.config.capacity_prob)

                self.opt_disc, self.grad_vars_disc, self.clipped_grad_vars_disc = \
                    get_opt(self.loss_disc, var_list=list(tf.trainable_variables('enc')+tf.trainable_variables('disc')), lr=self.lr_disc)
                
            else:
                self.loss = self.loss_recon + \
                                    self.beta * tf.maximum(self.loss_kl_sent_gmm, self.config.capacity_gmm) + \
                                    self.config.lam_disc * self.loss_disc_sent + \
                                    self.config.lam_disc * self.loss_disc_topic + \
                                    self.beta * tf.maximum(self.loss_kl_prob, self.config.capacity_prob)

                self.opt, self.grad_vars, self.clipped_grad_vars = \
                    get_opt(self.loss, var_list=tf.trainable_variables(), lr=self.lr, global_step=self.global_step)
                self.opt_infer, self.grad_vars_infer, self.clipped_grad_vars_infer = \
                    get_opt(self.loss, var_list=list(tf.trainable_variables('enc') + tf.trainable_variables('disc')), lr=self.lr, global_step=self.global_step)
                self.opt_gen, self.grad_vars_gen, self.clipped_grad_vars_gen = \
                    get_opt(self.loss, var_list=list(tf.trainable_variables('dec')), lr=self.lr, global_step=self.global_step)

                self.loss_disc = tf.constant(0, dtype=tf.float32)
                self.opt_disc = tf.constant(0, dtype=tf.float32)

            # for monitoring logdetcov
            self.logdetcovs_topic_posterior = tf_log(tf.linalg.det(self.covs_topic_posterior))
            self.mask_depth = tf.tile(tf.expand_dims(tf.constant([self.config.tree_depth[topic_idx]-1 for topic_idx in self.config.topic_idxs], \
                                                                 dtype=tf.int32), 0), [self.batch_l, 1])
            self.depth_logdetcovs_topic_posterior = tf.dynamic_partition(self.logdetcovs_topic_posterior, self.mask_depth, \
                                                                     num_partitions=max(self.config.tree_depth.values())) # list<depth> :n_topic_depth*batch_l
            
            # ------------------------------Evaluatiion------------------------------
            self.loss_list_train = [self.loss, self.loss_disc, self.loss_recon, self.loss_kl_prob, self.loss_kl_sent_gmm, self.loss_disc_sent, self.loss_disc_topic]
            self.loss_list_eval = [self.loss, self.loss_disc, self.loss_recon, self.loss_kl_prob, self.loss_kl_sent_gmm,  self.loss_disc_sent, self.loss_disc_topic, self.loss_kl_topic_gmm]
            self.loss_sum = (self.loss_recon + self.loss_kl_prob + self.loss_kl_sent_gmm) * self.n_sents
            
    def get_feed_dict(self, batch, mode, assertion=False):
        batch_l = len(batch)
        doc_l = np.array([instance.doc_l for instance in batch])
        max_doc_l = np.max(doc_l)
        sent_l = np.array([[instance.sent_l[i] if i < instance.doc_l else 0 for i in range(max_doc_l)] for instance in batch])
        max_sent_l = np.max(sent_l)
        dec_sent_l = np.array([[instance.sent_l[i] + 1 if i < instance.doc_l else 0 for i in range(max_doc_l)] for instance in batch])
        
        def token_dropout(sent_idxs, mode):
            sent_idxs_dropout = np.asarray(sent_idxs)
            if mode == 'train':
                word_keep_prob = self.config.word_keep_prob
            elif mode == 'eval':
                word_keep_prob = 1.
            sent_idxs_dropout[np.random.rand(len(sent_idxs)) > word_keep_prob] = self.config.UNK_IDX
            return list(sent_idxs_dropout)
                
        pad_token_idxs = lambda token_idxs_list, max_sent_l: np.array([pad_sequences(doc_token_idxs, maxlen=max_sent_l, padding='post', value=self.config.PAD_IDX, dtype=np.int32) for doc_token_idxs in token_idxs_list])
        
        enc_input_idxs_list = [[instance.token_idxs[i] if i < instance.doc_l else [] for i in range(max_doc_l)] for instance in batch]
        dec_input_idxs_list = [[[self.config.BOS_IDX] + token_dropout(sent_idxs, mode) for sent_idxs in doc_idxs] for doc_idxs in enc_input_idxs_list]
        dec_target_idxs_list = [[sent_idxs + [self.config.EOS_IDX] for sent_idxs in doc_idxs] for doc_idxs in enc_input_idxs_list]

        enc_input_idxs = pad_token_idxs(enc_input_idxs_list, max_sent_l)
        dec_input_idxs = pad_token_idxs(dec_input_idxs_list, max_sent_l+1)
        dec_target_idxs = pad_token_idxs(dec_target_idxs_list, max_sent_l+1)

        assert (batch_l, max_doc_l, max_sent_l) == enc_input_idxs.shape
        assert (batch_l, max_doc_l, max_sent_l+1) == dec_input_idxs.shape == dec_target_idxs.shape

        if mode == 'train':
            sample = True
            enc_keep_prob = self.config.enc_keep_prob
            dec_keep_prob = self.config.dec_keep_prob
        elif mode == 'eval':
            sample = False
            enc_keep_prob = dec_keep_prob = 1.

        feed_dict = {
                    self.t_variables['batch_l']: batch_l, self.t_variables['doc_l']: doc_l, self.t_variables['sent_l']: sent_l, self.t_variables['dec_sent_l']: dec_sent_l,
                    self.t_variables['enc_input_idxs']: enc_input_idxs, self.t_variables['dec_input_idxs']: dec_input_idxs, self.t_variables['dec_target_idxs']: dec_target_idxs, 
                    self.t_variables['enc_keep_prob']: enc_keep_prob, self.t_variables['dec_keep_prob']: dec_keep_prob, 
                    self.t_variables['sample']: sample
        }        
        return  feed_dict
