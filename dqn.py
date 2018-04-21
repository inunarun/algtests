"""
    :author: inunarun
             Aerospace Systems Design Laboratory,
             Georgia Institute of Technology,
             Atlanta, GA

    :date: 2018-04-20 16:42:06
"""

from layertypes import LayerTypes
import tensorflow as tf
from uuid import uuid4
import numpy as np


class DeepQNetwork(object):
    def __init__(self, num_states, num_actions, network_config,
                 learning_rate=0.01, gamma=0.95, epsilon_initial=1.0,
                 epsilon_final=0.01, epsilon_decay=0.999, batch_size=64,
                 target_update_freq=200, memory_size=1000000,
                 start_training_at=200, model_save_path="dqn.model"):
        super().__init__()

        self._gamma = gamma
        self._num_states = num_states
        self._batch_size = batch_size
        self._num_actions = num_actions
        self._memory_size = memory_size
        self._learning_rate = learning_rate
        self._epsilon_final = epsilon_final
        self._epsilon_decay = epsilon_decay
        self._epsilon_initial = epsilon_initial
        self._model_save_path = model_save_path
        self._start_training_at = start_training_at
        self._target_update_freq = target_update_freq
        self._memory = np.zeros((0, self._num_states * 2 + 2))

        self._epsilon = self._epsilon_initial

        self.__states = None
        self.__rewards = None
        self.__actions = None
        self.__nextstates = None

        self.__current_step = 0

        self.__session = tf.Session()

        self.__states, self.__qeval = self._build_network(network_config,
                                                          "eval")
        self.__nextstates, qnext = self._build_network(network_config,
                                                       "target")
        self.__loss, self.__training_op = self._setup_trainer(qnext)
        self.__target_update_op = self._define_network_update()

        self.__session.run(tf.global_variables_initializer())

        tf.summary.FileWriter("./logs/", self.__session.graph)

    @property
    def epsilon(self):
        return self._epsilon

    def observe(self, state, action, next_state, reward):
        # self._memory_size = (self._memory_size + 1)
        index = self._memory.shape[0] % self._memory_size
        transition = np.hstack((state, action, next_state, reward))
        self._memory = np.vstack((self._memory, transition))

    def predict(self, observation, custom_epsilon=None):
        if not custom_epsilon:
            epsilon = self._epsilon
        else:
            epsilon = custom_epsilon
        if np.random.uniform() < epsilon:
            observation = np.array(observation).reshape(1, -1)
            predicted_values = self.__session.run(self.__qeval,
                                                  feed_dict={
                                                    self.__states: observation
                                                  })
            action = np.argmax(predicted_values)
        else:
            action = np.random.randint(self._num_actions)

        return action

    def learn(self):
        if (self.__current_step + 1) % self._target_update_freq == 0:
            self.__session.run(self.__target_update_op)
            print("Successfully updated the target network.")

        if self._memory.shape[0] > self._start_training_at:
            indices = np.random.choice(self._memory.shape[0],
                                       size=self._batch_size)
            replay_memory = self._memory[indices]
            states = replay_memory[:, :self._num_states]
            actions = replay_memory[:, self._num_states]
            rewards = replay_memory[:, self._num_states + 1]
            next_states = replay_memory[:, -self._num_states:]

            _, cost = self.__session.run([self.__training_op, self.__loss],
                                         feed_dict={
                                            self.__states: states,
                                            self.__actions: actions,
                                            self.__rewards: rewards,
                                            self.__nextstates: next_states,
                                         })
            # print("Step: {0} | Training cost: {1}".format(self.__current_step,
            #                                               cost))

            # replace the epsilon calculation with an object.
            # tensorflow provides its own set of epsilon calculators.
            self._epsilon = max(self._epsilon * self._epsilon_decay,
                                 self._epsilon_final)
            self.__current_step += 1

    def _build_network(self, network_config, network_name):
        states = tf.placeholder(tf.float32, [None, self._num_states],
                                name="{0}_states".format(network_name))
        # next_states = tf.placeholder(tf.int32, [None, self._num_states],
        #                              name="next_states")
        # rewards = tf.placeholder(tf.float32, [None, ], name="rewards")
        # actions = tf.placeholder(tf.float32, [None, ], name="actions")

        # weights = tf.random_normal_initializer(0., 0.3)
        # biases = tf.constant_initializer(0.1)

        input_layer = states
        layer_count = 0
        with tf.variable_scope(network_name):
            for i, layer in enumerate(network_config):
                wgt_init = layer.weight_initializer
                bias_init = layer.bias_initializer
                if layer.name:
                    name = "{0}_{1}".format(network_name, name)
                    # name = layer.name
                else:
                    name = None
                    # name = uuid4().hex
                if layer.layer_type == LayerTypes.FULLY_CONNECTED:
                    input_layer = tf.layers.dense(input_layer,
                                                  layer.number_of_nodes,
                                                  activation=layer.activation,
                                                  use_bias=layer.use_bias,
                                                  kernel_initializer=wgt_init,
                                                  bias_initializer=bias_init,
                                                  name=name)
                else:
                    raise NotImplementedError("Not yet implemented.")

            weight_init = tf.random_normal_initializer(0., 0.3)
            bias_init = tf.constant_initializer(0.1)
            output_layer = tf.layers.dense(input_layer, self._num_actions,
                                           kernel_initializer=weight_init,
                                           bias_initializer=bias_init,
                                           name="qvalue")

        return states, output_layer

    def _setup_trainer(self, q_next):
        self.__rewards = tf.placeholder(tf.float32, [None, ], name="rewards")
        self.__actions = tf.placeholder(tf.int32, [None, ], name="actions")

        with tf.variable_scope("q_target"):
            q_target = tf.stop_gradient(self.__rewards +
                                        self._gamma *
                                        tf.reduce_max(q_next, axis=1))

        with tf.variable_scope("q_eval"):
            a_indices = tf.stack([tf.range(tf.shape(self.__actions)[0],
                                           dtype=tf.int32),
                                 self.__actions], axis=1)
            q_a_values = tf.gather_nd(params=self.__qeval, indices=a_indices)

        with tf.variable_scope("loss"):
            loss = tf.reduce_mean(tf.squared_difference(q_target, q_a_values,
                                                        name="TD_error"))

        with tf.variable_scope("train"):
            # make the definition of the optimizer configurable
            optimizer = tf.train.RMSPropOptimizer(self._learning_rate)
            training_operation = optimizer.minimize(loss)

        return loss, training_operation

    def _define_network_update(self):
        global_vars = tf.GraphKeys.GLOBAL_VARIABLES
        t_params = tf.get_collection(global_vars, scope="target")
        e_params = tf.get_collection(global_vars, scope="eval")
        update_op = [tf.assign(tp, ep) for tp, ep in zip(t_params, e_params)]
        return update_op
