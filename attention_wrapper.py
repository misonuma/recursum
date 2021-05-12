# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
A powerful dynamic attention wrapper object.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import functools
import math
import pdb

import numpy as np

from tensorflow.contrib.framework.python.framework import tensor_util
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import base as layers_base
from tensorflow.python.layers import core as layers_core
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import nest


__all__ = [
        "AttentionMechanism",
        "AttentionWrapper",
        "AttentionWrapperState",
        "BahdanauAttention"
]


_zero_state_tensors = rnn_cell_impl._zero_state_tensors    # pylint: disable=protected-access


class AttentionMechanism(object):

    @property
    def alignments_size(self):
        raise NotImplementedError

    @property
    def state_size(self):
        raise NotImplementedError


def _prepare_memory(memory, memory_sequence_length, check_inner_dims_defined):
    """Convert to tensor and possibly mask `memory`.

    Args:
        memory: `Tensor`, shaped `[batch_size, max_time, ...]`.
        memory_sequence_length: `int32` `Tensor`, shaped `[batch_size]`.
        check_inner_dims_defined: Python boolean.    If `True`, the `memory`
            argument's shape is checked to ensure all but the two outermost
            dimensions are fully defined.

    Returns:
        A (possibly masked), checked, new `memory`.

    Raises:
        ValueError: If `check_inner_dims_defined` is `True` and not
            `memory.shape[2:].is_fully_defined()`.
    """
    memory = nest.map_structure(
            lambda m: ops.convert_to_tensor(m, name="memory"), memory)
    if memory_sequence_length is not None:
        memory_sequence_length = ops.convert_to_tensor(
                memory_sequence_length, name="memory_sequence_length")
    if check_inner_dims_defined:
        def _check_dims(m):
            if not m.get_shape()[2:].is_fully_defined():
                raise ValueError("Expected memory %s to have fully defined inner dims, "
                                                 "but saw shape: %s" % (m.name, m.get_shape()))
        nest.map_structure(_check_dims, memory)
    if memory_sequence_length is None:
        seq_len_mask = None
    else:
        seq_len_mask = array_ops.sequence_mask(
                memory_sequence_length,
                maxlen=array_ops.shape(nest.flatten(memory)[0])[1],
                dtype=nest.flatten(memory)[0].dtype)
        seq_len_batch_size = (
                memory_sequence_length.shape[0].value
                or array_ops.shape(memory_sequence_length)[0])

    def _maybe_mask(m, seq_len_mask):
        rank = m.get_shape().ndims
        rank = rank if rank is not None else array_ops.rank(m)
        extra_ones = array_ops.ones(rank - 2, dtype=dtypes.int32)
        m_batch_size = m.shape[0].value or array_ops.shape(m)[0]
        if memory_sequence_length is not None:
            message = ("memory_sequence_length and memory tensor batch sizes do not "
                                 "match.")
            with ops.control_dependencies([
                    check_ops.assert_equal(
                            seq_len_batch_size, m_batch_size, message=message)]):
                seq_len_mask = array_ops.reshape(
                        seq_len_mask,
                        array_ops.concat((array_ops.shape(seq_len_mask), extra_ones), 0))
                return m * seq_len_mask
        else:
            return m
    return nest.map_structure(lambda m: _maybe_mask(m, seq_len_mask), memory)


def _maybe_mask_score(score, memory_sequence_length, score_mask_value):
    if memory_sequence_length is None:
        return score
    message = ("All values in memory_sequence_length must greater than zero.")
    with ops.control_dependencies(
            [check_ops.assert_positive(memory_sequence_length, message=message)]):
        score_mask = array_ops.sequence_mask(
                memory_sequence_length, maxlen=None)
        score_mask_values = score_mask_value * array_ops.ones_like(score)
        return array_ops.where(score_mask, score, score_mask_values)


class _BaseAttentionMechanism(AttentionMechanism):
    """A base AttentionMechanism class providing common functionality.

    Common functionality includes:
        1. Storing the query and memory layers.
        2. Preprocessing and storing the memory.
    """

    def __init__(self,
                     query_layer,
                     memory,
                     probability_fn,
                     memory_sequence_length=None,
                     memory_layer=None,
                     check_inner_dims_defined=True,
                     score_mask_value=None,
                     name=None):
        """Construct base AttentionMechanism class.

        Args:
            query_layer: Callable.    Instance of `tf.layers.Layer`.    The layer's depth
                must match the depth of `memory_layer`.    If `query_layer` is not
                provided, the shape of `query` must match that of `memory_layer`.
            memory: The memory to query; usually the output of an RNN encoder.    This
                tensor should be shaped `[batch_size, max_time, ...]`.
            probability_fn: A `callable`.    Converts the score and previous alignments
                to probabilities. Its signature should be:
                `probabilities = probability_fn(score, state)`.
            memory_sequence_length (optional): Sequence lengths for the batch entries
                in memory.    If provided, the memory tensor rows are masked with zeros
                for values past the respective sequence lengths.
            memory_layer: Instance of `tf.layers.Layer` (may be None).    The layer's
                depth must match the depth of `query_layer`.
                If `memory_layer` is not provided, the shape of `memory` must match
                that of `query_layer`.
            check_inner_dims_defined: Python boolean.    If `True`, the `memory`
                argument's shape is checked to ensure all but the two outermost
                dimensions are fully defined.
            score_mask_value: (optional): The mask value for score before passing into
                `probability_fn`. The default is -inf. Only used if
                `memory_sequence_length` is not None.
            name: Name to use when creating ops.
        """
        if (query_layer is not None
                and not isinstance(query_layer, layers_base.Layer)):
            raise TypeError(
                    "query_layer is not a Layer: %s" % type(query_layer).__name__)
        if (memory_layer is not None
                and not isinstance(memory_layer, layers_base.Layer)):
            raise TypeError(
                    "memory_layer is not a Layer: %s" % type(memory_layer).__name__)
        self._query_layer = query_layer
        self._memory_layer = memory_layer
        self.dtype = memory_layer.dtype
        if not callable(probability_fn):
            raise TypeError("probability_fn must be callable, saw type: %s" %
                                            type(probability_fn).__name__)
        if score_mask_value is None:
            score_mask_value = dtypes.as_dtype(
                    self._memory_layer.dtype).as_numpy_dtype(-np.inf)
#         self._probability_fn = lambda score, prev: (    # pylint:disable=g-long-lambda
#                 probability_fn(
#                         _maybe_mask_score(score, memory_sequence_length, score_mask_value),
#                         prev))
        self._probability_fn = probability_fn
        with ops.name_scope(
                name, "BaseAttentionMechanismInit", nest.flatten(memory)):
            self._values = _prepare_memory(
                    memory, memory_sequence_length,
                    check_inner_dims_defined=check_inner_dims_defined)
            self._keys = (
                    self.memory_layer(self._values) if self.memory_layer    # pylint: disable=not-callable
                    else self._values)
            self._batch_size = (
                    self._keys.shape[0].value or array_ops.shape(self._keys)[0])
            self._alignments_size = (self._keys.shape[1].value or
                                                             array_ops.shape(self._keys)[1])

    @property
    def memory_layer(self):
        return self._memory_layer

    @property
    def query_layer(self):
        return self._query_layer

    @property
    def values(self):
        return self._values

    @property
    def keys(self):
        return self._keys

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def alignments_size(self):
        return self._alignments_size

    @property
    def state_size(self):
        return self._alignments_size

    def initial_alignments(self, batch_size, dtype):
        """Creates the initial alignment values for the `AttentionWrapper` class.

        This is important for AttentionMechanisms that use the previous alignment
        to calculate the alignment at the next time step (e.g. monotonic attention).

        The default behavior is to return a tensor of all zeros.

        Args:
            batch_size: `int32` scalar, the batch_size.
            dtype: The `dtype`.

        Returns:
            A `dtype` tensor shaped `[batch_size, alignments_size]`
            (`alignments_size` is the values' `max_time`).
        """
        max_time = self._alignments_size
        return _zero_state_tensors(max_time, batch_size, dtype)

    def initial_state(self, batch_size, dtype):
        """Creates the initial state values for the `AttentionWrapper` class.

        This is important for AttentionMechanisms that use the previous alignment
        to calculate the alignment at the next time step (e.g. monotonic attention).

        The default behavior is to return the same output as initial_alignments.

        Args:
            batch_size: `int32` scalar, the batch_size.
            dtype: The `dtype`.

        Returns:
            A structure of all-zero tensors with shapes as described by `state_size`.
        """
        return self.initial_alignments(batch_size, dtype)

def _luong_score(query, keys, scale, state=None):
    """Implements Luong-style (multiplicative) scoring function.
    This attention has two forms.    The first is standard Luong attention,
    as described in:
    Minh-Thang Luong, Hieu Pham, Christopher D. Manning.
    "Effective Approaches to Attention-based Neural Machine Translation."
    EMNLP 2015.    https://arxiv.org/abs/1508.04025
    The second is the scaled form inspired partly by the normalized form of
    Bahdanau attention.
    To enable the second form, call this function with `scale=True`.
    Args:
        query: Tensor, shape `[batch_size, num_units]` to compare to keys.
        keys: Processed memory, shape `[batch_size, max_time, num_units]`.
        scale: the optional tensor to scale the attention score.
    Returns:
        A `[batch_size, max_time]` tensor of unnormalized score values.
    Raises:
        ValueError: If `key` and `query` depths do not match.
    """
    dtype = query.dtype
    depth = query.get_shape()[-1]
    key_units = keys.get_shape()[-1]
    if depth != key_units:
        raise ValueError(
                "Incompatible or unknown inner dimensions between query and keys.    "
                "Query (%s) has units: %s.    Keys (%s) have units: %s.    "
                "Perhaps you need to set num_units to the keys' dimension (%s)?" %
                (query, depth, keys, key_units, key_units))

    # Reshape from [batch_size, depth] to [batch_size, 1, depth]
    # for matmul.
    query = array_ops.expand_dims(query, 1)

    # Inner product along the query units dimension.
    # matmul shapes: query is [batch_size, 1, depth] and
    #                                keys is [batch_size, max_time, depth].
    # the inner product is asked to **transpose keys' inner shape** to get a
    # batched matmul on:
    #     [batch_size, 1, depth] . [batch_size, depth, max_time]
    # resulting in an output shape of:
    #     [batch_size, 1, max_time].
    # we then squeeze out the center singleton dimension.
    score = math_ops.matmul(query, keys, transpose_b=True)
    score = array_ops.squeeze(score, [1])

    if state is not None:
        c = variable_scope.get_variable("attention_c", dtype=dtype, shape=())
        score += math_ops.multiply(c, state) #     [batch_size, max_time]
    
    if scale is not None:
        score = scale * score
    return score


# +
class LuongAttention(_BaseAttentionMechanism):
    """Implements Luong-style (multiplicative) attention scoring.
    This attention has two forms.    The first is standard Luong attention,
    as described in:
    Minh-Thang Luong, Hieu Pham, Christopher D. Manning.
    [Effective Approaches to Attention-based Neural Machine Translation.
    EMNLP 2015.](https://arxiv.org/abs/1508.04025)
    The second is the scaled form inspired partly by the normalized form of
    Bahdanau attention.
    To enable the second form, construct the object with parameter
    `scale=True`.
    """

    def __init__(self,
                             num_units,
                             memory,
                             score_mask,
                             scale=False,
                             probability_fn=None,
                             score_mask_value=None,
                             coverage=False,
                             dtype=None,
                             custom_key_value_fn=None,
                             name="LuongAttention"):
        """Construct the AttentionMechanism mechanism.
        Args:
            num_units: The depth of the attention mechanism.
            memory: The memory to query; usually the output of an RNN encoder.    This
                tensor should be shaped `[batch_size, max_time, ...]`.
            memory_sequence_length: (optional) Sequence lengths for the batch entries
                in memory.    If provided, the memory tensor rows are masked with zeros
                for values past the respective sequence lengths.
            scale: Python boolean.    Whether to scale the energy term.
            probability_fn: (optional) A `callable`.    Converts the score to
                probabilities.    The default is `tf.nn.softmax`. Other options include
                `tf.contrib.seq2seq.hardmax` and `tf.contrib.sparsemax.sparsemax`.
                Its signature should be: `probabilities = probability_fn(score)`.
            score_mask_value: (optional) The mask value for score before passing into
                `probability_fn`. The default is -inf. Only used if
                `memory_sequence_length` is not None.
            dtype: The data type for the memory layer of the attention mechanism.
            custom_key_value_fn: (optional): The custom function for
                computing keys and values.
            name: Name to use when creating ops.
        """
        # For LuongAttention, we only transform the memory layer; thus
        # num_units **must** match expected the query depth.
        if score_mask_value is None:
            score_mask_value = dtypes.as_dtype(memory.dtype).as_numpy_dtype(-np.inf)
        
        if probability_fn is None:
            probability_fn = nn_ops.softmax
        if dtype is None:
            dtype = dtypes.float32
#         wrapped_probability_fn = lambda score, _: probability_fn(score)
        wrapped_probability_fn = lambda score, _: (    # pylint:disable=g-long-lambda
            probability_fn(self._maybe_mask_score(score, score_mask, score_mask_value)))        
        
#         super(LuongAttention, self).__init__(
        super().__init__(
                query_layer=None,
                memory_layer=layers_core.Dense(
                        num_units, name="memory_layer", use_bias=False, dtype=dtype),
                memory=memory,
                probability_fn=wrapped_probability_fn,
                score_mask_value=score_mask_value,
                name=name)
        self._num_units = num_units
        self._scale = scale
        self._coverage = coverage
        self._name = name

    def __call__(self, query, state):
        """Score the query based on the keys and values.
        Args:
            query: Tensor of dtype matching `self.values` and shape `[batch_size,
                query_depth]`.
            state: Tensor of dtype matching `self.values` and shape `[batch_size,
                alignments_size]` (`alignments_size` is memory's `max_time`).
        Returns:
            alignments: Tensor of dtype matching `self.values` and shape
                `[batch_size, alignments_size]` (`alignments_size` is memory's
                `max_time`).
        """
        with variable_scope.variable_scope(None, "luong_attention", [query]):
            attention_g = None
            if self._scale:
                attention_g = variable_scope.get_variable(
                        "attention_g",
                        dtype=query.dtype,
                        initializer=init_ops.ones_initializer,
                        shape=())
            if self._coverage:
                if state is None: raise
                score = _luong_score(query, self._keys, attention_g, state)
            else:
                score = _luong_score(query, self._keys, attention_g)
            
        alignments = self._probability_fn(score, state)
        next_state = alignments
        return alignments, next_state
    
    def _maybe_mask_score(self, score, score_mask, score_mask_value):
        score_mask_values = score_mask_value * array_ops.ones_like(score)
        return array_ops.where(score_mask, score, score_mask_values)


# -

class AttentionWrapperState(
        collections.namedtuple("AttentionWrapperState",
                                                     ("cell_state", "attention", "time", "alignments",
                                                        "alignment_history", "attention_state", "prob_gen"))):
    """`namedtuple` storing the state of a `AttentionWrapper`.

    Contains:

        - `cell_state`: The state of the wrapped `RNNCell` at the previous time
            step.
        - `attention`: The attention emitted at the previous time step.
        - `time`: int32 scalar containing the current time step.
        - `alignments`: A single or tuple of `Tensor`(s) containing the alignments
             emitted at the previous time step for each attention mechanism.
        - `alignment_history`: (if enabled) a single or tuple of `TensorArray`(s)
             containing alignment matrices from all time steps for each attention
             mechanism. Call `stack()` on each to convert to a `Tensor`.
        - `attention_state`: A single or tuple of nested objects
             containing attention mechanism state for each attention mechanism.
             The objects may contain Tensors or TensorArrays.
    """

    def clone(self, **kwargs):
        """Clone this object, overriding components provided by kwargs.

        The new state fields' shape must match original state fields' shape. This
        will be validated, and original fields' shape will be propagated to new
        fields.

        Example:

        ```python
        initial_state = attention_wrapper.zero_state(dtype=..., batch_size=...)
        initial_state = initial_state.clone(cell_state=encoder_state)
        ```

        Args:
            **kwargs: Any properties of the state object to replace in the returned
                `AttentionWrapperState`.

        Returns:
            A new `AttentionWrapperState` whose properties are the same as
            this one, except any overridden properties as provided in `kwargs`.
        """
        def with_same_shape(old, new):
            """Check and set new tensor's shape."""
            if isinstance(old, ops.Tensor) and isinstance(new, ops.Tensor):
                return tensor_util.with_same_shape(old, new)
            return new

        return nest.map_structure(
                with_same_shape,
                self,
                super(AttentionWrapperState, self)._replace(**kwargs))


def _compute_attention(attention_mechanism, cell_output, attention_state, attention_layer):
    """Computes the attention and alignments for a given attention_mechanism."""
    alignments, next_attention_state = attention_mechanism(cell_output, state=attention_state)

    # Reshape from [batch_size, memory_time] to [batch_size, 1, memory_time]
    expanded_alignments = array_ops.expand_dims(alignments, 1)
    # Context is the inner product of alignments and values along the
    # memory time dimension.
    # alignments shape is
    #     [batch_size, 1, memory_time]
    # attention_mechanism.values shape is
    #     [batch_size, memory_time, memory_size]
    # the batched matmul is over memory_time, so the output shape is
    #     [batch_size, 1, memory_size].
    # we then squeeze out the singleton dim.
    context = math_ops.matmul(expanded_alignments, attention_mechanism.values)
    context = array_ops.squeeze(context, [1])

    if attention_layer is not None:
        attention = attention_layer(array_ops.concat([cell_output, context], 1))
    else:
        attention = context

    return attention, alignments, next_attention_state

class AttentionWrapper(rnn_cell_impl.RNNCell):
    """Wraps another `RNNCell` with attention.
    """

    def __init__(self,
                             cell,
                             attention_mechanism,
                             attention_layer_size=None,
                             alignment_history=False,
                             cell_input_fn=None,
                             output_attention=True,
                             initial_cell_state=None,
                             name=None,
                             attention_layer=None,
                             prob_gen_layer=None):
        """Construct the `AttentionWrapper`.

        **NOTE** If you are using the `BeamSearchDecoder` with a cell wrapped in
        `AttentionWrapper`, then you must ensure that:

        - The encoder output has been tiled to `beam_width` via
            `tf.contrib.seq2seq.tile_batch` (NOT `tf.tile`).
        - The `batch_size` argument passed to the `zero_state` method of this
            wrapper is equal to `true_batch_size * beam_width`.
        - The initial state created with `zero_state` above contains a
            `cell_state` value containing properly tiled final state from the
            encoder.

        An example:

        ```
        tiled_encoder_outputs = tf.contrib.seq2seq.tile_batch(
                encoder_outputs, multiplier=beam_width)
        tiled_encoder_final_state = tf.conrib.seq2seq.tile_batch(
                encoder_final_state, multiplier=beam_width)
        tiled_sequence_length = tf.contrib.seq2seq.tile_batch(
                sequence_length, multiplier=beam_width)
        attention_mechanism = MyFavoriteAttentionMechanism(
                num_units=attention_depth,
                memory=tiled_inputs,
                memory_sequence_length=tiled_sequence_length)
        attention_cell = AttentionWrapper(cell, attention_mechanism, ...)
        decoder_initial_state = attention_cell.zero_state(
                dtype, batch_size=true_batch_size * beam_width)
        decoder_initial_state = decoder_initial_state.clone(
                cell_state=tiled_encoder_final_state)
        ```

        Args:
            cell: An instance of `RNNCell`.
            attention_mechanism: A list of `AttentionMechanism` instances or a single
                instance.
            attention_layer_size: A list of Python integers or a single Python
                integer, the depth of the attention (output) layer(s). If None
                (default), use the context as attention at each time step. Otherwise,
                feed the context and cell output into the attention layer to generate
                attention at each time step. If attention_mechanism is a list,
                attention_layer_size must be a list of the same length. If
                attention_layer is set, this must be None.
            alignment_history: Python boolean, whether to store alignment history
                from all time steps in the final output state (currently stored as a
                time major `TensorArray` on which you must call `stack()`).
            cell_input_fn: (optional) A `callable`.    The default is:
                `lambda inputs, attention: array_ops.concat([inputs, attention], -1)`.
            output_attention: Python bool.    If `True` (default), the output at each
                time step is the attention value.    This is the behavior of Luong-style
                attention mechanisms.    If `False`, the output at each time step is
                the output of `cell`.    This is the behavior of Bhadanau-style
                attention mechanisms.    In both cases, the `attention` tensor is
                propagated to the next time step via the state and is used there.
                This flag only controls whether the attention mechanism is propagated
                up to the next cell in an RNN stack or to the top RNN output.
            initial_cell_state: The initial state value to use for the cell when
                the user calls `zero_state()`.    Note that if this value is provided
                now, and the user uses a `batch_size` argument of `zero_state` which
                does not match the batch size of `initial_cell_state`, proper
                behavior is not guaranteed.
            name: Name to use when creating ops.
            attention_layer: A list of `tf.layers.Layer` instances or a
                single `tf.layers.Layer` instance taking the context and cell output as
                inputs to generate attention at each time step. If None (default), use
                the context as attention at each time step. If attention_mechanism is a
                list, attention_layer must be a list of the same length. If
                attention_layers_size is set, this must be None.

        Raises:
            TypeError: `attention_layer_size` is not None and (`attention_mechanism`
                is a list but `attention_layer_size` is not; or vice versa).
            ValueError: if `attention_layer_size` is not None, `attention_mechanism`
                is a list, and its length does not match that of `attention_layer_size`;
                if `attention_layer_size` and `attention_layer` are set simultaneously.
        """
#         super(AttentionWrapper, self).__init__(name=name)
        super().__init__(name=name)
        rnn_cell_impl.assert_like_rnncell("cell", cell)
        if isinstance(attention_mechanism, (list, tuple)):
            self._is_multi = True
            attention_mechanisms = attention_mechanism
#             for attention_mechanism in attention_mechanisms:
#                 if not isinstance(attention_mechanism, AttentionMechanism):
#                     raise TypeError(
#                             "attention_mechanism must contain only instances of "
#                             "AttentionMechanism, saw type: %s"
#                             % type(attention_mechanism).__name__)
        else:
            self._is_multi = False
#             if not isinstance(attention_mechanism, AttentionMechanism):
#                 raise TypeError(
#                         "attention_mechanism must be an AttentionMechanism or list of "
#                         "multiple AttentionMechanism instances, saw type: %s"
#                         % type(attention_mechanism).__name__)
            attention_mechanisms = (attention_mechanism,)

        if cell_input_fn is None:
            cell_input_fn = (
                    lambda inputs, attention: array_ops.concat([inputs, attention], -1))
        else:
            if not callable(cell_input_fn):
                raise TypeError(
                        "cell_input_fn must be callable, saw type: %s"
                        % type(cell_input_fn).__name__)

        if attention_layer_size is not None and attention_layer is not None:
            raise ValueError("Only one of attention_layer_size and attention_layer "
                                             "should be set")

        if attention_layer_size is not None:
            attention_layer_sizes = tuple(
                    attention_layer_size
                    if isinstance(attention_layer_size, (list, tuple))
                    else (attention_layer_size,))
            if len(attention_layer_sizes) != len(attention_mechanisms):
                raise ValueError(
                        "If provided, attention_layer_size must contain exactly one "
                        "integer per attention_mechanism, saw: %d vs %d"
                        % (len(attention_layer_sizes), len(attention_mechanisms)))
            self._attention_layers = tuple(
                    layers_core.Dense(
                            attention_layer_size,
                            name="attention_layer",
                            use_bias=False,
                            dtype=attention_mechanisms[i].dtype)
                    for i, attention_layer_size in enumerate(attention_layer_sizes))
            self._attention_layer_size = sum(attention_layer_sizes)
        elif attention_layer is not None:
            self._attention_layers = tuple(
                    attention_layer
                    if isinstance(attention_layer, (list, tuple))
                    else (attention_layer,))
            if len(self._attention_layers) != len(attention_mechanisms):
                raise ValueError(
                        "If provided, attention_layer must contain exactly one "
                        "layer per attention_mechanism, saw: %d vs %d"
                        % (len(self._attention_layers), len(attention_mechanisms)))
            self._attention_layer_size = sum(
                    layer.compute_output_shape(
                            [None,
                             cell.output_size + mechanism.values.shape[-1].value])[-1].value
                    for layer, mechanism in zip(
                            self._attention_layers, attention_mechanisms))
        else:
            self._attention_layers = None
            self._attention_layer_size = sum(
                    attention_mechanism.values.get_shape()[-1].value
                    for attention_mechanism in attention_mechanisms)

        self._cell = cell
        self._attention_mechanisms = attention_mechanisms
        self._cell_input_fn = cell_input_fn
        self._output_attention = output_attention
        self._alignment_history = alignment_history
        self._prob_gen_layer = prob_gen_layer
        
        with ops.name_scope(name, "AttentionWrapperInit"):
            if initial_cell_state is None:
                self._initial_cell_state = None
            else:
                final_state_tensor = nest.flatten(initial_cell_state)[-1]
                state_batch_size = (
                        final_state_tensor.shape[0].value
                        or array_ops.shape(final_state_tensor)[0])
                error_message = (
                        "When constructing AttentionWrapper %s: " % self._base_name +
                        "Non-matching batch sizes between the memory "
                        "(encoder output) and initial_cell_state.    Are you using "
                        "the BeamSearchDecoder?    You may need to tile your initial state "
                        "via the tf.contrib.seq2seq.tile_batch function with argument "
                        "multiple=beam_width.")
                with ops.control_dependencies(
                        self._batch_size_checks(state_batch_size, error_message)):
                    self._initial_cell_state = nest.map_structure(
                            lambda s: array_ops.identity(s, name="check_initial_cell_state"),
                            initial_cell_state)

    def _batch_size_checks(self, batch_size, error_message):
        return [check_ops.assert_equal(batch_size, attention_mechanism.batch_size, message=error_message)
                                                                    for attention_mechanism in self._attention_mechanisms]

    def _item_or_tuple(self, seq):
        """Returns `seq` as tuple or the singular element.

        Which is returned is determined by how the AttentionMechanism(s) were passed
        to the constructor.

        Args:
            seq: A non-empty sequence of items or generator.

        Returns:
             Either the values in the sequence as a tuple if AttentionMechanism(s)
             were passed to the constructor as a sequence or the singular element.
        """
        t = tuple(seq)
        if self._is_multi:
            return t
        else:
            return t[0]

    @property
    def output_size(self):
        if self._output_attention:
            return self._attention_layer_size
        else:
            return self._cell.output_size

    @property
    def state_size(self):
        """The `state_size` property of `AttentionWrapper`.

        Returns:
            An `AttentionWrapperState` tuple containing shapes used by this object.
        """
        return AttentionWrapperState(
                cell_state=self._cell.state_size,
                time=tensor_shape.TensorShape([]),
                attention=self._attention_layer_size,
                alignments=self._item_or_tuple(
                        a.alignments_size for a in self._attention_mechanisms),
                attention_state=self._item_or_tuple(
                        a.state_size for a in self._attention_mechanisms),
                alignment_history=self._item_or_tuple(
                        a.alignments_size if self._alignment_history else ()
                        for a in self._attention_mechanisms),
                prob_gen=tensor_shape.TensorShape([1]))    # sometimes a TensorArray

    def zero_state(self, batch_size, dtype):
        """Return an initial (zero) state tuple for this `AttentionWrapper`.

        **NOTE** Please see the initializer documentation for details of how
        to call `zero_state` if using an `AttentionWrapper` with a
        `BeamSearchDecoder`.

        Args:
            batch_size: `0D` integer tensor: the batch size.
            dtype: The internal state data type.

        Returns:
            An `AttentionWrapperState` tuple containing zeroed out tensors and,
            possibly, empty `TensorArray` objects.

        Raises:
            ValueError: (or, possibly at runtime, InvalidArgument), if
                `batch_size` does not match the output size of the encoder passed
                to the wrapper object at initialization time.
        """
        with ops.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
            if self._initial_cell_state is not None:
                cell_state = self._initial_cell_state
            else:
                cell_state = self._cell.zero_state(batch_size, dtype)
            error_message = (
                    "When calling zero_state of AttentionWrapper %s: " % self._base_name +
                    "Non-matching batch sizes between the memory "
                    "(encoder output) and the requested batch size.    Are you using "
                    "the BeamSearchDecoder?    If so, make sure your encoder output has "
                    "been tiled to beam_width via tf.contrib.seq2seq.tile_batch, and "
                    "the batch_size= argument passed to zero_state is "
                    "batch_size * beam_width.")
#             with ops.control_dependencies(
#                     self._batch_size_checks(batch_size, error_message)):
            cell_state = nest.map_structure(
                    lambda s: array_ops.identity(s, name="checked_cell_state"),
                    cell_state)
            initial_alignments = [
                    attention_mechanism.initial_alignments(batch_size, dtype)
                    for attention_mechanism in self._attention_mechanisms]
            return AttentionWrapperState(
                    cell_state=cell_state,
                    time=array_ops.zeros([], dtype=dtypes.int32),
                    attention=_zero_state_tensors(self._attention_layer_size, batch_size,
                                                                                dtype),
                    alignments=self._item_or_tuple(initial_alignments),
                    attention_state=self._item_or_tuple(
                            attention_mechanism.initial_state(batch_size, dtype)
                            for attention_mechanism in self._attention_mechanisms),
                    alignment_history=self._item_or_tuple(
                            tensor_array_ops.TensorArray(
                                    dtype,
                                    size=0,
                                    dynamic_size=True,
                                    clear_after_read=False,
                                    element_shape=alignment.shape)
                            if self._alignment_history else ()
                            for alignment in initial_alignments),
                    prob_gen=array_ops.ones([batch_size, 1], dtype=dtype))

    def call(self, inputs, state):
        """Perform a step of attention-wrapped RNN.

        - Step 1: Mix the `inputs` and previous step's `attention` output via
            `cell_input_fn`.
        - Step 2: Call the wrapped `cell` with this input and its previous state.
        - Step 3: Score the cell's output with `attention_mechanism`.
        - Step 4: Calculate the alignments by passing the score through the
            `normalizer`.
        - Step 5: Calculate the context vector as the inner product between the
            alignments and the attention_mechanism's values (memory).
        - Step 6: Calculate the attention output by concatenating the cell output
            and context through the attention layer (a linear layer with
            `attention_layer_size` outputs).

        Args:
            inputs: (Possibly nested tuple of) Tensor, the input at this time step.
            state: An instance of `AttentionWrapperState` containing
                tensors from the previous time step.

        Returns:
            A tuple `(attention_or_cell_output, next_state)`, where:

            - `attention_or_cell_output` depending on `output_attention`.
            - `next_state` is an instance of `AttentionWrapperState`
                 containing the state calculated at this time step.

        Raises:
            TypeError: If `state` is not an instance of `AttentionWrapperState`.
        """
        if not isinstance(state, AttentionWrapperState):
            raise TypeError("Expected state to be instance of AttentionWrapperState. "
                                            "Received type %s instead."    % type(state))

        # Step 1: Calculate the true inputs to the cell based on the
        # previous attention value.
        cell_inputs = self._cell_input_fn(inputs, state.attention)
        cell_state = state.cell_state
        cell_output, next_cell_state = self._cell(cell_inputs, cell_state)
        cell_batch_size = (cell_output.shape[0].value or array_ops.shape(cell_output)[0])

        cell_output = array_ops.identity(
                cell_output, name="checked_cell_output")

        if self._is_multi:
            previous_attention_state = state.attention_state
            previous_alignment_history = state.alignment_history
        else:
            previous_attention_state = [state.attention_state]
            previous_alignment_history = [state.alignment_history]

        all_alignments = []
        all_attentions = []
        all_attention_states = []
        maybe_all_histories = []
        for i, attention_mechanism in enumerate(self._attention_mechanisms):
            if self._attention_mechanisms[i]._coverage:
                coverage_state = control_flow_ops.cond(math_ops.equal(previous_alignment_history[i].size(), 0), \
                                                             lambda: array_ops.zeros_like(state.alignments), \
                                                             lambda: math_ops.reduce_sum(previous_alignment_history[i].stack(), [0]))
                attention, alignments, next_attention_state = _compute_attention(
                        attention_mechanism, cell_output, coverage_state,
                        self._attention_layers[i] if self._attention_layers else None)
            else:
                attention, alignments, next_attention_state = _compute_attention(
                        attention_mechanism, cell_output, previous_attention_state[i],
                        self._attention_layers[i] if self._attention_layers else None)
                
            alignment_history = previous_alignment_history[i].write(
                    state.time, alignments) if self._alignment_history else ()

            all_attention_states.append(next_attention_state)
            all_alignments.append(alignments)
            all_attentions.append(attention)
            maybe_all_histories.append(alignment_history)

        attention = array_ops.concat(all_attentions, 1)
        
        if self._prob_gen_layer is not None:
            prob_gen = self._prob_gen_layer(self._cell_input_fn(inputs, attention))
        else:
            prob_gen = array_ops.ones([cell_batch_size, 1], dtype=dtypes.float32)
        
        next_state = AttentionWrapperState(
                time=state.time+1,
                cell_state=next_cell_state,
                attention=attention,
                attention_state=self._item_or_tuple(all_attention_states),
                alignments=self._item_or_tuple(all_alignments),
                alignment_history=self._item_or_tuple(maybe_all_histories),
                prob_gen=prob_gen)

        if self._output_attention:
            return attention, next_state
        else:
            return cell_output, next_state
