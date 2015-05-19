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


def generate_preprocessor(size, gray_scale):
    resamp = resampler(size)

    def preprocessor(state):
        return resamp(gray_scale(state))

    return preprocessor


class ExpGain(object):
    def __init__(self, net, actions, preprocessor, game, dataset, init_state):
        self.net = net                      # dqn action selector
        self.actions = actions              # list of actions
        self.preprocessor = preprocessor    # downsampler
        self.game = game                    # game updater
        self.dataset = dataset              # replay_dataset object
        self.sequence = deque()             # sequence of frames
        self.init_state = init_state        # initial state
        for _ in range(_NFRAME):
            self.sequence.append(init_state)

    def reset_game(self):
        self.sequence = deque()
        for _ in range(_NFRAME):
            self.sequence.append(self.init_state)

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

        exp_action = self.actions.index(action)
        exp_frame = self.preprocessor(self.arrayify_frames())
        self.dataset.add_experience(exp_action, reward, exp_frame)
        self.record(exp_action, reward, exp_frame)

        if reward < 0:  # game over
            self.reset_game()

    # TODO: decide *.out file directory
    def record(self, action, reward, frame):
        with open("actions.out", "a") as actions:
            actions.write(str(action)+'\n')

        with open("rewards.out", "a") as rewards:
            rewards.write(str(reward)+'\n')

        with open("frames.out", "a") as frames:
            frames.write(str(frame)+'\n')
