"""
    :author: inunarun
             Aerospace Systems Design Laboratory,
             Georgia Institute of Technology,
             Atlanta, GA

    :date: 2018-04-20 18:01:36
"""

from neuralnetwork import NeuralNetwork
from discrete_env import Maze
from dqn import DeepQNetwork
import tensorflow as tf
import numpy as np

class Controller(object):
    def __init__(self, num_episodes, num_steps_per_episode):
        super().__init__()
        self.__num_episodes = num_episodes
        self.__num_steps_per_episode = num_steps_per_episode

    def train(self, env, algorithm):
        for episode in range(self.__num_episodes):
            # env.set_directory("./episodes/{0}".format(episode))
            episode_reward, step_number = 0, 0
            state = env.reset()
            start_state = state.copy()
            while True:
                env.render()
                action = algorithm.predict(state)
                next_state, reward, done = env.step(action)
                algorithm.observe(state, action, next_state, reward)
                algorithm.learn()
                step_number += 1

                episode_reward += reward

                if step_number > self.__num_steps_per_episode or done:
                    break

            print("Episode: {0} | Number of Steps: {2} | Reward: {1}"
                  " | Epsilon: {3} | Memory: {4} | Start at: {5}"
                  "".format(episode + 1, episode_reward, step_number,
                            algorithm.epsilon, algorithm._memory.shape[0],
                            start_state))
        print("Completed training.")

    def play(self, env, algorithm):
        pass

if __name__ == '__main__':
    maze = Maze(5, 5, num_bombs=1)

    weight_init = tf.random_normal_initializer(0., 0.3)
    bias_init = tf.constant_initializer(0.1)
    network_config = NeuralNetwork()
    # network_config.add_new_dense_layer(30, tf.nn.relu,
    #                                    weight_initializer=weight_init,
    #                                    bias_initializer=bias_init)
    network_config.add_new_dense_layer(30, tf.nn.relu,
                                       weight_initializer=weight_init,
                                       bias_initializer=bias_init)

    algorithm = DeepQNetwork(maze.n_states, maze.n_actions, network_config,
                             epsilon_initial=0.1, epsilon_decay=1.0)

    controller = Controller(1000000, 50)
    controller.train(maze, algorithm)
