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

_weights_stdev = None

def set_weights_stdev(weights_stdev):
    global _weights_stdev
    _weights_stdev = weights_stdev

def unset_weights_stdev():
    global _weights_stdev
    _weights_stdev = None

def Deconv2D(
    name, 
    input_dim, 
    output_dim, 
    filter_size, 
    inputs, 
    he_init=True,
    weightnorm=None,
    biases=True,
    gain=1.,
    mask_type=None,
):
    """
    inputs: tensor of shape (batch size, height, width, input_dim)
    returns: tensor of shape (batch size, 2*height, 2*width, output_dim)
    """
    with tf.name_scope(name):

        if mask_type is not None:
            raise Exception('Unsupported configuration')

        def uniform(stdev, size):
            return np.random.uniform(
                low=-stdev * np.sqrt(3),
                high=stdev * np.sqrt(3),
                size=size
            ).astype('float32')

        stride = 2
        fan_in = input_dim * filter_size**2 // (stride**2)  # Fixed integer division
        fan_out = output_dim * filter_size**2

        if he_init:
            filters_stdev = np.sqrt(4. / (fan_in + fan_out))
        else:  # Normalized init (Glorot & Bengio)
            filters_stdev = np.sqrt(2. / (fan_in + fan_out))

        if _weights_stdev is not None:
            filter_values = uniform(
                _weights_stdev,
                (filter_size, filter_size, output_dim, input_dim)
            )
        else:
            filter_values = uniform(
                filters_stdev,
                (filter_size, filter_size, output_dim, input_dim)
            )

        filter_values *= gain

        filters = lib.param(
            name + '.Filters',
            filter_values
        )

        if weightnorm is None:  # Fixed comparison (was == None)
            weightnorm = _default_weightnorm
        if weightnorm:
            norm_values = np.sqrt(np.sum(np.square(filter_values), axis=(0, 1, 3)))
            target_norms = lib.param(
                name + '.g',
                norm_values
            )
            with tf.name_scope('weightnorm'):
                norms = tf.sqrt(tf.reduce_sum(tf.square(filters), axis=[0, 1, 3]))  # Fixed TensorFlow API
                filters = filters * tf.expand_dims(target_norms / norms, 1)

        inputs = tf.transpose(inputs, [0, 2, 3, 1], name='NCHW_to_NHWC')

        input_shape = tf.shape(inputs)
        
        # Fixed tf.pack() -> tf.stack() for TensorFlow 1.x+ compatibility
        output_shape = tf.stack([input_shape[0], 2 * input_shape[1], 2 * input_shape[2], output_dim])

        result = tf.nn.conv2d_transpose(
            input=inputs, 
            filters=filters,
            output_shape=output_shape, 
            strides=[1, 2, 2, 1],
            padding='SAME'
        )

        if biases:
            _biases = lib.param(
                name + '.Biases',
                np.zeros(output_dim, dtype='float32')
            )
            result = tf.nn.bias_add(result, _biases)

        result = tf.transpose(result, [0, 3, 1, 2], name='NHWC_to_NCHW')

        return result
