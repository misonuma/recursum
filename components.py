#coding:utf-8
import tensorflow as tf
import numpy as np
import pdb

CLIP = 1e-8

def tf_log(x):
    return tf.log(tf.clip_by_value(x, CLIP, x))

def tf_clip_vals(vals, probs):
    vals_clipped = tf.where(tf.equal(probs, 0), tf.zeros_like(vals, dtype=tf.float32), vals)
    return vals_clipped

def tf_clip_means(means, probs):
    means_zero = tf.zeros_like(means, dtype=tf.float32)
    means_clipped = tf.where(tf.equal(tf.tile(tf.expand_dims(probs, -1), [1, 1, tf.shape(means)[-1]]), 0.), means_zero, means)
    return means_clipped

def tf_clip_covs(covs, probs):
    covs_eye = tf.eye(tf.shape(covs)[-2], tf.shape(covs)[-1], batch_shape=[tf.shape(covs)[0], tf.shape(covs)[1]], dtype=tf.float32)
    covs_clipped = tf.where(tf.equal(tf.tile(tf.expand_dims(tf.expand_dims(probs, -1), -1), [1, 1, tf.shape(covs)[-2], tf.shape(covs)[-1]]), 0.), covs_eye, covs)
    return covs_clipped

def sample_latents(means, logvars, seed, sample):
    noises = tf.cond(sample, lambda: tf.random.normal(tf.shape(means), seed=seed, dtype=tf.float32), lambda: tf.zeros_like(means, dtype=tf.float32))
        
    # reparameterize
    latents = means + tf.exp(0.5 * logvars) * noises
    return latents

def sample_latents_fullcov(means, covs, seed, sample):
    noises = tf.cond(sample, lambda: tf.random.normal(tf.shape(means), seed=seed, dtype=tf.float32), lambda: tf.zeros_like(means, dtype=tf.float32))
        
    # reparameterize
    scales = tf.linalg.cholesky(covs)
    latents = means + tf.squeeze(tf.matmul(scales, tf.expand_dims(noises, -1)), -1)
    return latents

def sample_gumbels(logits, temperature, seed, sample):
    noises = tf.cond(sample, \
                     lambda: -tf_log(-tf_log(tf.random.uniform(tf.shape(logits), minval=0., maxval=1., seed=seed, dtype=tf.float32))), \
                     lambda: tf.zeros_like(logits, dtype=tf.float32))
    
    # reparameterize
    latents = tf.nn.softmax((logits+noises)/temperature, -1)
    return latents

# get topic prior distribution for each document
def get_params_topic_prior(model, params_topic_posterior, param_root_prior):
    tree_params_topic_prior = {}
    tree_params_topic_prior[0] = tf.expand_dims(param_root_prior, 1)
    for child_idx, parent_idx in model.config.child_to_parent_idxs.items():
        tree_params_topic_prior[child_idx] = tf.expand_dims(params_topic_posterior[:, model.config.topic_idxs.index(parent_idx), :], 1)
    params_topic_prior = tf.concat([tree_params_topic_prior[topic_idx] for topic_idx in model.config.topic_idxs], 1)
    return params_topic_prior

def compute_kl_losses(means, logvars, means_prior=None, logvars_prior=None):
    if means_prior is None and logvars_prior is None:
        kl_losses = tf.reduce_sum(-0.5 * (logvars - tf.square(means) - tf.exp(logvars) + 1.0), -1) # sum over latent dimension    
    elif means_prior is not None and logvars_prior is None:
        kl_losses = tf.reduce_sum(-0.5 * (logvars - tf.square(means-means_prior) - tf.exp(logvars) + 1.0), -1) # sum over latent dimension 
    else:
        kl_losses= 0.5 * tf.reduce_sum(tf.exp(logvars-logvars_prior) + tf.square(means_prior - means) / tf.clip_by_value(tf.exp(logvars_prior), CLIP, tf.exp(logvars_prior)) - 1 + (logvars_prior - logvars), -1) # sum over latent dimension    
    return kl_losses

def compute_kl_losses_sent_gauss(model):
    diff_topic_posterior = model.means_topic_posterior - model.means_topic_prior # batch_l x n_topic x dim_latent
    add_covs_topic_posterior = model.covs_topic_posterior + tf.matmul(tf.expand_dims(diff_topic_posterior, -1), tf.expand_dims(diff_topic_posterior, -2)) # batch_l x n_topicx dim_latent x dim_latent
    diag_covs_topic = tf.expand_dims(tf.matrix_diag_part(tf.linalg.solve(model.covs_topic_prior, add_covs_topic_posterior)), 1) # batch_l x 1 x n_topic x dim_latent

    logdet_covs_topic_prior = 2.*tf.expand_dims(tf_log(tf.matrix_diag_part(tf.linalg.cholesky(model.covs_topic_prior))), 1) # batch_l x 1 x n_topic x dim_latent
    logdet_covs_sent_posterior = tf.expand_dims(model.logvars_sent_posterior, 2) # batch_l x max_doc_l x 1 x dim_latent

    losses_kl_sent_gauss = 0.5 * tf.reduce_sum(logdet_covs_topic_prior - logdet_covs_sent_posterior + diag_covs_topic - 1., -1) # batch_l x max_doc_l x n_topic, sum over latent dimension
    return losses_kl_sent_gauss

def compute_kl_losses_topic_gauss(model):
    diff_topic_posterior = model.means_topic_posterior - model.means_topic_prior # batch_l x n_topic x dim_latent
    add_covs_topic_posterior = model.covs_topic_posterior + tf.matmul(tf.expand_dims(diff_topic_posterior, -1), tf.expand_dims(diff_topic_posterior, -2)) # batch_l x n_topic x dim_latent x dim_latent
    diag_covs_topic = tf.expand_dims(tf.matrix_diag_part(tf.linalg.solve(model.covs_topic_prior, add_covs_topic_posterior)), 1) # batch_l x 1 x n_topic x dim_latent

    logdet_covs_topic_prior = 2.*tf.expand_dims(tf_log(tf.matrix_diag_part(tf.linalg.cholesky(model.covs_topic_prior))), 1) # batch_l x 1 x n_topic x dim_latent
    logdet_covs_topic_posterior = 2.*tf.expand_dims(tf_log(tf.matrix_diag_part(tf.linalg.cholesky(model.covs_topic_posterior))), 1) # batch_l x 1 x n_topic x dim_latent

    losses_kl_topic_gauss = 0.5 * tf.reduce_sum(logdet_covs_topic_prior - logdet_covs_topic_posterior + diag_covs_topic - 1., -1) # batch_l x 1 x n_topic,  sum over latent dimension
    return losses_kl_topic_gauss

def compute_kl_losses_topic_paris_gauss(model):
    diff_topic_posterior = tf.expand_dims(model.means_topic_posterior, 1) - tf.expand_dims(model.means_topic_posterior, 2) # batch_l x n_topic x n_topic x dim_latent
    add_covs_topic_posterior = tf.expand_dims(model.covs_topic_posterior, 2) + tf.matmul(tf.expand_dims(diff_topic_posterior, -1), tf.expand_dims(diff_topic_posterior, -2)) # batch_l x n_topic x n_topic x dim_latent x dim_latent
#     diag_covs_topic = tf.matrix_diag_part(tf.linalg.solve(tf.tile(tf.expand_dims(model.covs_topic_posterior, 2), [1, 1, tf.shape(add_covs_topic_posterior)[2], 1, 1]), add_covs_topic_posterior)) # batch_l x n_topic x n_topic x dim_latent
    diag_covs_topic = tf.matrix_diag_part(tf.linalg.solve(tf.tile(tf.expand_dims(model.covs_topic_posterior, 1), [1, tf.shape(add_covs_topic_posterior)[1], 1, 1, 1]), add_covs_topic_posterior)) # batch_l x n_topic x n_topic x dim_latent

    logdet_covs_topic_posterior = 2.*tf_log(tf.matrix_diag_part(tf.linalg.cholesky(model.covs_topic_posterior))) # batch_l x n_topic x dim_latent
    
    losses_kl_topic_pairs_gauss = tf.expand_dims(0.5 * tf.reduce_sum(tf.expand_dims(logdet_covs_topic_posterior, 1) - tf.expand_dims(logdet_covs_topic_posterior, 2) + diag_covs_topic - 1., -1), 1) # batch_l x 1 x n_topic x n_topic
    return losses_kl_topic_pairs_gauss


def compute_losses_coverage(model, dec_final_state, dec_mask_sent, n_tiled):
    alignments = dec_final_state.alignment_history.stack()
    coverages = tf.scan(lambda a, x: a + x, alignments)
    losses_sent_coverage = tf.reshape(tf.transpose(tf.reduce_sum(tf.minimum(alignments, coverages), -1)), [model.batch_l, n_tiled, tf.shape(dec_mask_sent)[-1]])
    losses_coverage = tf.reduce_sum(losses_sent_coverage*dec_mask_sent, -1)
    return losses_coverage
