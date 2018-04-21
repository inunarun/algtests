"""
    :author: inunarun
             Aerospace Systems Design Laboratory,
             Georgia Institute of Technology,
             Atlanta, GA

    :date: 2018-04-20 16:49:09
"""

import matplotlib.pyplot as plt
import numpy as np
import os


class Maze(object):
    def __init__(self, num_rows, num_cols, num_bombs=1, directory=None):
        assert(num_rows > 2 and num_cols > 2)

        super().__init__()
        self.n_actions = 4
        self.n_states = 2

        self._current_position = [-1, -1]

        self.__directory = directory
        self.__num_rows = num_rows
        self.__num_cols = num_cols
        self.__viewer = None
        self.__axis = None
        self.__step_id = 0

        self._bombs_ind = None
        self._target_ind = None
        self._current_ind = None

        target_x = np.random.randint(self.__num_rows)
        target_y = np.random.randint(self.__num_cols)
        self.__target = [target_x, target_y]

        self.__bombs = []
        for bomb_i in range(num_bombs):
            bomb_pos = self.__target.copy()
            while bomb_pos == self.__target:
                bomb_x = np.random.randint(self.__num_rows)
                bomb_y = np.random.randint(self.__num_cols)
                bomb_pos = [bomb_x, bomb_y]
            self.__bombs.append(bomb_pos)

        # self.__generate_plot()

    @staticmethod
    def name():
        return "2D Maze with Bombs"

    def set_directory(self, directory):
        self.__directory = directory
        if not os.path.exists(directory):
            os.makedirs(directory)

    def reset(self):
        self.__step_id = 0
        self._current_position = self.__target.copy()
        while self._current_position == self.__target:
            row_position = np.random.randint(self.__num_rows)
            col_position = np.random.randint(self.__num_cols)

            self._current_position = [row_position, col_position]

        self.render()

        return self._current_position

    def step(self, action):
        if action == 0:
            # move up
            action = np.array([0, 1])
        elif action == 1:
            # move right
            action = np.array([1, 0])
        elif action == 2:
            # move down
            action = np.array([0, -1])
        elif action == 3:
            # move left
            action = np.array([-1, 0])
        else:
            raise ValueError("Improper action value, %s" % action)

        position = np.array(self._current_position) + action
        self._current_position = position.tolist()
        if (position[0] < 0 or position[0] >= self.__num_cols or
            position[1] < 0 or position[1] >= self.__num_rows):
            fail = True
            done = True
        elif self._current_position in self.__bombs:
            fail = True
            done = True
        elif self._current_position == self.__target:
            fail = False
            done = True
        else:
            fail = False
            done = False

        if done and not fail:
            reward = 1
        elif done and fail:
            reward = -1
        else:
            reward = 0

        self.__step_id += 1

        self.render()

        return self._current_position, reward, done

    def render(self):
        pass
        # self._current_ind.set_offsets(self._current_position)
        # self.__viewer.canvas.draw()
        # if self.__directory:
        #     fig_path = os.path.join(self.__directory, "figure_%s.png" %
        #                             self.__step_id)
        #     self.__viewer.savefig(fig_path)

    def __generate_plot(self):
        self.__viewer = plt.figure()
        self.__axis = self.__viewer.add_axes([0.05, 0.05, 0.9, 0.9])

        self.__axis.set_xlim(-1, self.__num_cols)
        self.__axis.set_ylim(-1, self.__num_rows)

        x, y = np.meshgrid(np.arange(0, self.__num_cols),
                           np.arange(0, self.__num_rows))
        positions = np.vstack([x.ravel(), y.ravel()])

        self.__axis.scatter(positions[0, :], positions[1, :], marker="*",
                            color="b")
        bombs = np.array(self.__bombs)
        self._bombs_ind = self.__axis.scatter(bombs[:, 0], bombs[:, 1],
                                              marker="o", color="k")
        self._current_ind = self.__axis.scatter(self._current_position[0],
                                                self._current_position[1],
                                                marker="^", color="r")
        self._target_ind = self.__axis.scatter(self.__target[0],
                                               self.__target[1],
                                               marker="v", color="g")
        # if not self.__directory:
        #     self.__viewer.show()
