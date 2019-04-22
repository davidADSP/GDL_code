

import tensorflow as tf
import keras

from keras.layers import Layer, InputSpec
import keras.backend as K

class ReflectionPadding2D(Layer):
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        self.input_spec = [InputSpec(ndim=4)]
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, s):
        """ If you are using "channels_last" configuration"""
        return (s[0], s[1] + 2 * self.padding[0], s[2] + 2 * self.padding[1], s[3])

    def call(self, x, mask=None):
        w_pad,h_pad = self.padding
        return tf.pad(x, [[0,0], [h_pad,h_pad], [w_pad,w_pad], [0,0] ], 'REFLECT')



class RepeatVector4D(Layer):

    def __init__(self, n, **kwargs):
        self.n = n
        self.input_spec = [InputSpec(ndim=3)]
        super(RepeatVector4D, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.n, input_shape[1], input_shape[2])

    def call(self, x, mask=None):
        x = K.expand_dims(x, 1)
        pattern = K.stack([1, self.n, 1, 1])
        return K.tile(x, pattern)



