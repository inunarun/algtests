"""
    :author: inunarun
             Aerospace Systems Design Laboratory,
             Georgia Institute of Technology,
             Atlanta, GA

    :date: 2018-04-20 18:31:10
"""

from layer import NeuralNetworkLayer as Layer
from layertypes import LayerTypes
import tensorflow as tf



class NeuralNetwork(list):
    def add_new_dense_layer(self, num_units, activation, use_bias=True,
                            name=None, bias_initializer=tf.zeros_initializer(),
                            weight_initializer=None):
        new_layer = Layer(num_units, activation, LayerTypes.FULLY_CONNECTED,
                          use_bias, weight_initializer, bias_initializer,
                          name)
        self.append(new_layer)

    def append(self, entry):
        assert(isinstance(entry, Layer))
        super().append(entry)
