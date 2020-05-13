from BanditAlgorithm import BanditAlgorithm
from utils import update_mean_inc, update_var_inc

import numpy as np
import random

class UCB(BanditAlgorithm):
    def __init__(self):
        return

    def initialize_vars(self):
        self.n_choice = np.zeros(self.n_arms, dtype=int)
        if self.optimistic_factor is None:
            self.sample_mean = np.zeros(self.n_arms)
        elif isinstance(self.optimistic_factor, float) or isinstance(self.optimistic_factor, int):
            self.sample_mean = np.ones(self.n_arms) * self.optimistic_factor
        self.play_once_flag = False
        self.timestep = 0

    def __init__(self, n_arms, const_c, optimistic_factor=None):
        self.n_arms = n_arms
        self.arms = np.arange(n_arms)
        self.optimistic_factor = optimistic_factor
        self.const_c = const_c
        self.initialize_vars()
        return

    def reset(self):
        del(self.n_choice, self.sample_mean)
        self.initialize_vars()
        return

    def set_problem(self, means, stds):
        self.means = means
        self.stds = stds
        return

    def get_max_arm(self):
        return np.argmax(self.sample_mean)

    def play_once(self):
        locs = np.argwhere(self.n_choice == 0)
        if len(locs) > 1:
            return locs[0][0]
        elif len(locs) == 1:
            self.play_once_flag = True
            return locs[0][0]

    def select_arm(self, timestep):
        if self.play_once_flag is True:
            lnt = np.log(timestep)
            ucb = self.sample_mean + (self.const_c * np.sqrt(lnt * (1.0 / self.n_choice)))
            return np.argmax(ucb)
        else:
            return self.play_once()


    def update(self, arm, reward):
        self.n_choice[arm] += 1
        self.sample_mean[arm] = update_mean_inc(self.sample_mean[arm], reward, self.n_choice[arm])

    def play(self, timestep):
        chosen_arm = self.select_arm(timestep)
        return chosen_arm





