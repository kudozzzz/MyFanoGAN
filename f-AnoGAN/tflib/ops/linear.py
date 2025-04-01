"""
Copyright (c) 2017 Ishaan Gulrajani

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import tflib as lib
import numpy as np
import tensorflow as tf

_default_weightnorm = False

def enable_default_weightnorm():
    global _default_weightnorm
    _default_weightnorm = True

def disable_default_weightnorm():
    global _default_weightnorm
    _default_weightnorm = False

_weights_stdev = None

def set_weights_stdev(weights_stdev):
    global _weights_stdev
    _weights_stdev = weights_stdev

def unset_weights_stdev():
    global _weights_stdev
    _weights_stdev = None

def Linear(
        name, 
        input_dim, 
        output_dim, 
        inputs,
        biases=True,
        initialization=None,
        weightnorm=None,
        gain=1.
    ):
    """
    initialization: None, `lecun`, 'glorot', `he`, 'glorot_he', `orthogonal`, `("uniform", range)`
    """
    def uniform(stdev, size):
        if _weights_stdev is not None:
            stdev = _weights_stdev
        return np.random.uniform(
            low=-stdev * np.sqrt(3),
            high=stdev * np.sqrt(3),
            size=size
        ).astype('float32')

    if initialization == 'lecun': 
        weight_values = uniform(np.sqrt(1./input_dim), (input_dim, output_dim))

    elif initialization in ['glorot', None]:
        weight_values = uniform(np.sqrt(2./(input_dim + output_dim)), (input_dim, output_dim))

    elif initialization == 'he':
        weight_values = uniform(np.sqrt(2./input_dim), (input_dim, output_dim))

    elif initialization == 'glorot_he':
        weight_values = uniform(np.sqrt(4./(input_dim + output_dim)), (input_dim, output_dim))

    elif initialization == 'orthogonal' or (initialization is None and input_dim == output_dim):
        def sample(shape):
            if len(shape) < 2:
                raise RuntimeError("Only shapes of length 2 or more are supported.")
            flat_shape = (shape[0], np.prod(shape[1:]))
            a = np.random.normal(0.0, 1.0, flat_shape)
            u, _, v = np.linalg.svd(a, full_matrices=False)
            q = u if u.shape == flat_shape else v
            return q.reshape(shape).astype('float32')

        weight_values = sample((input_dim, output_dim))

    elif isinstance(initialization, tuple) and initialization[0] == 'uniform':
        weight_values = np.random.uniform(
            low=-initialization[1],
            high=initialization[1],
            size=(input_dim, output_dim)
        ).astype('float32')

    else:
        raise Exception('Invalid initialization!')

    weight_values *= gain
    weight = lib.param(name + '.W', weight_values)

    if weightnorm is None:
        weightnorm = _default_weightnorm

    if weightnorm:
        norm_values = np.sqrt(np.sum(np.square(weight_values), axis=0))
        target_norms = lib.param(name + '.g', norm_values)
        
        norms = tf.sqrt(tf.reduce_sum(tf.square(weight), axis=0))  # Changed `reduction_indices` to `axis`
        weight = weight * (target_norms / norms)

    if inputs.get_shape().ndims == 2:
        result = tf.matmul(inputs, weight)
    else:
        reshaped_inputs = tf.reshape(inputs, [-1, input_dim])
        result = tf.matmul(reshaped_inputs, weight)
        result = tf.reshape(result, tf.stack(tf.unstack(tf.shape(inputs))[:-1] + [output_dim]))  # Updated `tf.pack()` and `tf.unpack()`

    if biases:
        result = tf.nn.bias_add(
            result,
            lib.param(name + '.b', np.zeros((output_dim,), dtype='float32'))
        )

    return result
