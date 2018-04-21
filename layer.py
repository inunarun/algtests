"""
    :author: inunarun
             Aerospace Systems Design Laboratory,
             Georgia Institute of Technology,
             Atlanta, GA

    :date: 2018-04-20 18:34:10
"""


class NeuralNetworkLayer(object):
    def __init__(self, num_nodes, activation, layer_type, use_bias,
                 weight_initializer, bias_initializer, name):
        super().__init__()

        self._weight_init = weight_initializer
        self._bias_init = bias_initializer
        self._activation = activation
        self._layer_type = layer_type
        self._num_nodes = num_nodes
        self._uses_bias = use_bias
        self._name = name

    @property
    def layer_type(self):
        return self._layer_type

    @property
    def number_of_nodes(self):
        return self._num_nodes

    @property
    def activation(self):
        return self._activation

    @property
    def name(self):
        return self._name

    @property
    def use_bias(self):
        return self._uses_bias

    @property
    def weight_initializer(self):
        return self._weight_init

    @property
    def bias_initializer(self):
        return self._bias_init
