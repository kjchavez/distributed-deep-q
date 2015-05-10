import numpy as np
import random
from collections import deque
import scipy.ndimage

_FRAME_LIMIT = 1000000
_EPSILON_MAX = 1.0
_EPSILON_MIN = 0.1
_NFRAME = 4


def resampler(size):
    def func(state):
        zoom_factor = (1., float(size[0])/state.shape[1],
                       float(size[1])/state.shape[2])
        return scipy.ndimage.zoom(state, zoom_factor, order=0)

    return func


class ExpGain(object):
    def __init__(self, net, actions, preprocessor, game, dataset, init_state):
        self.net = net                      # dqn action selector
        self.actions = actions              # list of actions
        self.preprocessor = preprocessor    # downsampler
        self.game = game                    # game updater
        self.dataset = dataset              # replay_dataset object
        self.sequence = deque()             # sequence of frames
        for _ in range(_NFRAME):
            self.sequence.append(init_state)

    def select_action(self, pstate, epsilon):
        if random.random() < epsilon:
            return random.choice(self.actions)
        else:
            return self.actions[self.net.select_action(pstate)]

    def get_epsilon(self, iter_num):
        if iter_num > _FRAME_LIMIT:
            return _EPSILON_MIN
        else:
            return _EPSILON_MIN + (_EPSILON_MAX - _EPSILON_MIN) \
                   * (_FRAME_LIMIT - iter_num) / _FRAME_LIMIT

    def arrayify_frames(self):
        nx, ny = self.sequence[0].shape
        array = np.zeros((_NFRAME, ny, nx), dtype='uint8')
        for frame in range(_NFRAME):
            array[frame] = self.sequence[frame]
        return array

    def generate_experience(self, iter_num):
        pstate = self.preprocessor(self.arrayify_frames())
        action = self.select_action(pstate, self.get_epsilon(iter_num))
        new_state, reward = self.game(self.sequence[-1], action)
        self.sequence.popleft()
        self.sequence.append(new_state)
        self.dataset.add_experience(
            self.actions.index(action), reward,
            self.preprocessor(self.arrayify_frames())
        )
