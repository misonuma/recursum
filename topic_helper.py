# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A library of helpers for use with SamplingDecoders.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import six
import numpy as np

from tensorflow.contrib.seq2seq.python.ops import decoder
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sort_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops.distributions import bernoulli
from tensorflow.python.ops.distributions import categorical
from tensorflow.python.util import nest

from components import sample_gumbels

__all__ = [
    "Helper",
    "TrainingHelper",
    "SampleEmbeddingHelper",
    "GumbelSoftmaxEmbeddingHelper",
]

_transpose_batch_time = decoder._transpose_batch_time    # pylint: disable=protected-access


def _call_sampler(sample_n_fn, sample_shape, name=None):
    """Reshapes vector of samples."""
    with ops.name_scope(name, "call_sampler", values=[sample_shape]):
        sample_shape = ops.convert_to_tensor(
                sample_shape, dtype=dtypes.int32, name="sample_shape")
        # Ensure sample_shape is a vector (vs just a scalar).
        pad = math_ops.cast(math_ops.equal(array_ops.rank(sample_shape), 0),
                                                dtypes.int32)
        sample_shape = array_ops.reshape(
                sample_shape,
                array_ops.pad(array_ops.shape(sample_shape),
                                            paddings=[[pad, 0]],
                                            constant_values=1))
        samples = sample_n_fn(math_ops.reduce_prod(sample_shape))
        batch_event_shape = array_ops.shape(samples)[1:]
        final_shape = array_ops.concat([sample_shape, batch_event_shape], 0)
        return array_ops.reshape(samples, final_shape)


def categorical_sample(logits, dtype=dtypes.int32,
                                             sample_shape=(), seed=None):
    """Samples from categorical distribution."""
    logits = ops.convert_to_tensor(logits, name="logits")
    event_size = array_ops.shape(logits)[-1]
    batch_shape_tensor = array_ops.shape(logits)[:-1]
    def _sample_n(n):
        """Sample vector of categoricals."""
        if logits.shape.ndims == 2:
            logits_2d = logits
        else:
            logits_2d = array_ops.reshape(logits, [-1, event_size])
        sample_dtype = dtypes.int64 if logits.dtype.size > 4 else dtypes.int32
        draws = random_ops.multinomial(
                logits_2d, n, seed=seed, output_dtype=sample_dtype)
        draws = array_ops.reshape(
                array_ops.transpose(draws),
                array_ops.concat([[n], batch_shape_tensor], 0))
        return math_ops.cast(draws, dtype)
    return _call_sampler(_sample_n, sample_shape)


def _unstack_ta(inp):
    return tensor_array_ops.TensorArray(
            dtype=inp.dtype, size=array_ops.shape(inp)[0],
            element_shape=inp.get_shape()[1:]).unstack(inp)


@six.add_metaclass(abc.ABCMeta)
class Helper(object):
    """Interface for implementing sampling in seq2seq decoders.

    Helper instances are used by `BasicDecoder`.
    """

    @abc.abstractproperty
    def batch_size(self):
        """Batch size of tensor returned by `sample`.

        Returns a scalar int32 tensor.
        """
        raise NotImplementedError("batch_size has not been implemented")

    @abc.abstractproperty
    def sample_ids_shape(self):
        """Shape of tensor returned by `sample`, excluding the batch dimension.

        Returns a `TensorShape`.
        """
        raise NotImplementedError("sample_ids_shape has not been implemented")

    @abc.abstractproperty
    def sample_ids_dtype(self):
        """DType of tensor returned by `sample`.

        Returns a DType.
        """
        raise NotImplementedError("sample_ids_dtype has not been implemented")

    @abc.abstractmethod
    def initialize(self, name=None):
        """Returns `(initial_finished, initial_inputs)`."""
        pass

    @abc.abstractmethod
    def sample(self, time, outputs, state, name=None):
        """Returns `sample_ids`."""
        pass

    @abc.abstractmethod
    def next_inputs(self, time, outputs, state, sample_ids, name=None):
        """Returns `(finished, next_inputs, next_state)`."""
        pass

class TrainingHelper(Helper):
    """A helper for use during training.    Only reads inputs.

    Returned sample_ids are the argmax of the RNN output logits.
    """

    def __init__(self, inputs, sequence_length, time_major=False, name=None):
        """Initializer.

        Args:
            inputs: A (structure of) input tensors.
            sequence_length: An int32 vector tensor.
            time_major: Python bool.    Whether the tensors in `inputs` are time major.
                If `False` (default), they are assumed to be batch major.
            name: Name scope for any created operations.

        Raises:
            ValueError: if `sequence_length` is not a 1D tensor.
        """
        with ops.name_scope(name, "TrainingHelper", [inputs, sequence_length]):
            inputs = ops.convert_to_tensor(inputs, name="inputs")
            self._inputs = inputs
            if not time_major:
                inputs = nest.map_structure(_transpose_batch_time, inputs)

            self._input_tas = nest.map_structure(_unstack_ta, inputs)
            self._sequence_length = ops.convert_to_tensor(
                    sequence_length, name="sequence_length")
            if self._sequence_length.get_shape().ndims != 1:
                raise ValueError(
                        "Expected sequence_length to be a vector, but received shape: %s" %
                        self._sequence_length.get_shape())

            self._zero_inputs = nest.map_structure(
                    lambda inp: array_ops.zeros_like(inp[0, :]), inputs)

            self._batch_size = array_ops.size(sequence_length)

    @property
    def inputs(self):
        return self._inputs

    @property
    def sequence_length(self):
        return self._sequence_length

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def sample_ids_shape(self):
        return tensor_shape.TensorShape([])

    @property
    def sample_ids_dtype(self):
        return dtypes.int32

    def initialize(self, name=None):
        with ops.name_scope(name, "TrainingHelperInitialize"):
            finished = math_ops.equal(0, self._sequence_length)
            all_finished = math_ops.reduce_all(finished)
            next_inputs = control_flow_ops.cond(
                    all_finished, lambda: self._zero_inputs,
                    lambda: nest.map_structure(lambda inp: inp.read(0), self._input_tas))
            return (finished, next_inputs)

    def sample(self, time, outputs, name=None, **unused_kwargs):
        with ops.name_scope(name, "TrainingHelperSample", [time, outputs]):
            sample_ids = math_ops.cast(
                    math_ops.argmax(outputs, axis=-1), dtypes.int32)
            return sample_ids

    def next_inputs(self, time, outputs, state, name=None, **unused_kwargs):
        """next_inputs_fn for TrainingHelper."""
        with ops.name_scope(name, "TrainingHelperNextInputs",
                                                [time, outputs, state]):
            next_time = time + 1
            finished = (next_time >= self._sequence_length)
            all_finished = math_ops.reduce_all(finished)
            def read_from_ta(inp):
                return inp.read(next_time)
            next_inputs = control_flow_ops.cond(
                    all_finished, lambda: self._zero_inputs,
                    lambda: nest.map_structure(read_from_ta, self._input_tas))
            return (finished, next_inputs, state)

class GreedyEmbeddingHelper(Helper):
    """A helper for use during inference.
    Uses the argmax of the output (treated as logits) and passes the
    result through an embedding layer to get the next input.
    """

    def __init__(self, embedding, start_tokens, end_token, latents_input=None):
        """Initializer.
        Args:
            embedding: A callable that takes a vector tensor of `ids` (argmax ids),
                or the `params` argument for `embedding_lookup`. The returned tensor
                will be passed to the decoder input.
            start_tokens: `int32` vector shaped `[batch_size]`, the start tokens.
            end_token: `int32` scalar, the token that marks end of decoding.
        Raises:
            ValueError: if `start_tokens` is not a 1D tensor or `end_token` is not a
                scalar.
        """
        if callable(embedding):
            self._embedding_fn = embedding
        else:
            self._embedding_fn = (
                    lambda ids: embedding_ops.embedding_lookup(embedding, ids))

        self._start_tokens = ops.convert_to_tensor(
                start_tokens, dtype=dtypes.int32, name="start_tokens")
        self._end_token = ops.convert_to_tensor(
                end_token, dtype=dtypes.int32, name="end_token")
        if self._start_tokens.get_shape().ndims != 1:
            raise ValueError("start_tokens must be a vector")
        self._batch_size = array_ops.size(start_tokens)
        if self._end_token.get_shape().ndims != 0:
            raise ValueError("end_token must be a scalar")
        self._start_inputs = self._embedding_fn(self._start_tokens)

        self._latents_input = latents_input
        if self._latents_input is not None: self._start_inputs = array_ops.concat([self._start_inputs, self._latents_input], 1)
        
    @property
    def batch_size(self):
        return self._batch_size

    @property
    def sample_ids_shape(self):
        return tensor_shape.TensorShape([])

    @property
    def sample_ids_dtype(self):
        return dtypes.int32

    def initialize(self, name=None):
        finished = array_ops.tile([False], [self._batch_size])
        return (finished, self._start_inputs)

    def sample(self, time, outputs, state, name=None):
        """sample for GreedyEmbeddingHelper."""
        del time, state    # unused by sample_fn
        # Outputs are logits, use argmax to get the most probable id
        if not isinstance(outputs, ops.Tensor):
            raise TypeError("Expected outputs to be a single Tensor, got: %s" %
                                            type(outputs))
        sample_ids = math_ops.argmax(outputs, axis=-1, output_type=dtypes.int32)
        return sample_ids

    def next_inputs(self, time, outputs, state, sample_ids, name=None):
        """next_inputs_fn for GreedyEmbeddingHelper."""
        del time, outputs    # unused by next_inputs_fn
        finished = math_ops.equal(sample_ids, self._end_token)
        all_finished = math_ops.reduce_all(finished)
        
        if self._latents_input is None:
                _next_inputs = self._embedding_fn(sample_ids)
        else:
                _next_inputs = array_ops.concat([self._embedding_fn(sample_ids), self._latents_input], 1)
        
        next_inputs = control_flow_ops.cond(
                all_finished,
                # If we're finished, the next_inputs value doesn't matter
                lambda: self._start_inputs,
                lambda: _next_inputs)
        return (finished, next_inputs, state)


class SampleEmbeddingHelper(GreedyEmbeddingHelper):
    """A helper for use during inference.
    Uses sampling (from a distribution) instead of argmax and passes the
    result through an embedding layer to get the next input.
    """

    def __init__(self, embedding, start_tokens, end_token,
                             nucleus=None, softmax_temperature=None, seed=None, latents_input=None):
        """Initializer.
        Args:
            embedding: A callable that takes a vector tensor of `ids` (argmax ids),
                or the `params` argument for `embedding_lookup`. The returned tensor
                will be passed to the decoder input.
            start_tokens: `int32` vector shaped `[batch_size]`, the start tokens.
            end_token: `int32` scalar, the token that marks end of decoding.
            softmax_temperature: (Optional) `float32` scalar, value to divide the
                logits by before computing the softmax. Larger values (above 1.0) result
                in more random samples, while smaller values push the sampling
                distribution towards the argmax. Must be strictly greater than 0.
                Defaults to 1.0.
            seed: (Optional) The sampling seed.
        Raises:
            ValueError: if `start_tokens` is not a 1D tensor or `end_token` is not a
                scalar.
        """
#         super(SampleEmbeddingHelper, self).__init__(
#                 embedding, start_tokens, end_token)
        super().__init__(embedding, start_tokens, end_token, latents_input)
        self._nucleus = nucleus
        self._softmax_temperature = softmax_temperature
        self._seed = seed

    def sample(self, time, outputs, state, name=None):
        """sample for SampleEmbeddingHelper."""
        del time, state    # unused by sample_fn
        # Outputs are logits, we sample instead of argmax (greedy).
        if not isinstance(outputs, ops.Tensor):
            raise TypeError("Expected outputs to be a single Tensor, got: %s" %
                                            type(outputs))
            
        probs = nn_ops.softmax(outputs, -1)
        sorted_args = sort_ops.argsort(probs, -1, direction='DESCENDING')
        sorted_nucleus_probs = math_ops.cumsum(sort_ops.sort(probs, -1, direction='DESCENDING'), -1) < self._nucleus
        nucleus_probs = array_ops.gather(sorted_nucleus_probs, sort_ops.argsort(sorted_args, -1, direction='ASCENDING'), batch_dims=1)
        argmax_probs = array_ops.one_hot(math_ops.argmax(outputs, -1), depth=array_ops.shape(outputs)[-1], on_value=True, off_value=False, dtype=dtypes.bool)
        outputs = array_ops.where((nucleus_probs|argmax_probs), outputs, -np.inf*array_ops.ones_like(outputs, dtype=dtypes.float32))
            
        if self._softmax_temperature is None:
            logits = outputs
        else:
            logits = outputs / self._softmax_temperature

        sample_ids = categorical_sample(logits=logits, seed=self._seed)

        return sample_ids


class GumbelSoftmaxEmbeddingHelper(Helper):
    """A helper for use during training.

    Uses the argmax of the output (treated as logits) and passes the
    result through an embedding layer to get the next input.
    """

    def __init__(self, embedding, start_tokens, end_token, softmax_temperature, seed, sample, latents_input=None):
        """Initializer.

        Args:
            embedding: A callable that takes a vector tensor of `ids` (argmax ids),
                or the `params` argument for `embedding_lookup`. The returned tensor
                will be passed to the decoder input.
            start_tokens: `int32` vector shaped `[batch_size]`, the start tokens.
            end_token: `int32` scalar, the token that marks end of decoding.

        Raises:
            ValueError: if `start_tokens` is not a 1D tensor or `end_token` is not a
                scalar.
        """
        
        if callable(embedding):
            self._embedding_fn = embedding
        else:
            self._embedding_fn = (
                    lambda tokens: math_ops.tensordot(tokens, embedding, axes=[[-1], [0]]))
            self._embedding_size = array_ops.shape(embedding)[0]

        self._start_tokens = ops.convert_to_tensor(
                start_tokens, dtype=dtypes.int32, name="start_tokens")
        self._end_token = ops.convert_to_tensor(
                end_token, dtype=dtypes.int32, name="end_token")
        if self._start_tokens.get_shape().ndims != 1:
            raise ValueError("start_tokens must be a vector")
        self._batch_size = array_ops.size(start_tokens)
        if self._end_token.get_shape().ndims != 0:
            raise ValueError("end_token must be a scalar")
        
        self._latents_input = latents_input
        soft_start_tokens = array_ops.one_hot(self._start_tokens, self._embedding_size, dtype=dtypes.float32)
        self._start_inputs = self._embedding_fn(soft_start_tokens)
        if self._latents_input is not None: self._start_inputs = array_ops.concat([self._start_inputs, self._latents_input], 1)
        
        self._softmax_temperature = softmax_temperature
        self._seed = seed
        self._sample = sample
        
    @property
    def batch_size(self):
        return self._batch_size

    @property
    def sample_ids_shape(self):
        emb = array_ops.zeros_like(self._embedding_size)
        return emb.get_shape()[:1]

    @property
    def sample_ids_dtype(self):
        return dtypes.float32

    def initialize(self, name=None):
        finished = array_ops.tile([False], [self._batch_size])
        return (finished, self._start_inputs)

    def sample(self, time, outputs, state, name=None):
        """sample for GreedyEmbeddingHelper."""
        del time, state    # unused by sample_fn
        # Outputs are logits, use argmax to get the most probable id
        if not isinstance(outputs, ops.Tensor):
            raise TypeError("Expected outputs to be a single Tensor, got: %s" %
                                            type(outputs))
        sample_tokens = sample_gumbels(outputs, self._softmax_temperature, self._seed, self._sample)
        return sample_tokens

    def next_inputs(self, time, outputs, state, sample_ids, name=None):
        """next_inputs_fn for GreedyEmbeddingHelper."""
        del time, outputs # unused by next_inputs_fn
        
        hard_ids = math_ops.argmax(sample_ids, axis=-1, output_type=dtypes.int32)
        finished = math_ops.equal(hard_ids, self._end_token)
        all_finished = math_ops.reduce_all(finished)
        
        if self._latents_input is None:
                _next_inputs = self._embedding_fn(sample_ids)
        else:
                _next_inputs = array_ops.concat([self._embedding_fn(sample_ids), self._latents_input], 1)

        next_inputs = control_flow_ops.cond(
                all_finished,
                # If we're finished, the next_inputs value doesn't matter
                lambda: self._start_inputs,
                lambda: _next_inputs)
        return (finished, next_inputs, state)
