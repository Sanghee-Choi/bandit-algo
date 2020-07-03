from algorithms.BanditAlgorithm import BanditAlgorithm
from utils import update_mean_inc

import numpy as np
import random

class EpsilonGreedy(BanditAlgorithm):
    def __init__(self):
        return

    def initialize_vars(self):
        self.n_choice = np.zeros(self.n_arms, dtype=int)
        if self.optimistic_factor is None:
            self.sample_mean = np.zeros(self.n_arms)
        elif isinstance(self.optimistic_factor, float) or isinstance(self.optimistic_factor, int):
            self.sample_mean = np.ones(self.n_arms) * self.optimistic_factor
        return

    def __init__(self, n_arms, epsilon, optimistic_factor=None):
        self.n_arms = n_arms
        self.arms = np.arange(n_arms)
        self.optimistic_factor = optimistic_factor
        self.epsilon = epsilon
        self.initialize_vars()
        self.name = 'eps-greedy_{:.5f}'.format(epsilon)
        return

    def reset(self):
        del(self.n_choice, self.sample_mean)
        self.initialize_vars()
        return

    def set_problem(self, means, stds):
        self.means = means
        self.stds = stds
        return

    def select_arm(self):
        prob = np.random.random()
        if prob < self.epsilon:
            return np.random.choice(self.arms)
        else:
            return np.argmax(self.sample_mean)
        return 

    def update(self, arm, reward):
        self.n_choice[arm] += 1
        self.sample_mean[arm] = update_mean_inc(
            self.sample_mean[arm], reward, self.n_choice[arm]
        )

    def play(self):
        chosen_arm = self.select_arm()
        return chosen_arm


class EpsilonGreedyNonstationary(EpsilonGreedy):
    def __init__(self, n_arms, epsilon, alpha, optimistic_factor=None):
        self.n_arms = n_arms
        self.arms = np.arange(n_arms)
        self.optimistic_factor = optimistic_factor
        self.epsilon = epsilon
        self.mean_update_factor = alpha
        self.initialize_vars()
        self.name = 'nonstationary_eps-greedy_{:.5f}'.format(epsilon)
        #print(self.mean_update_factor)
        return

    def update(self, arm, reward):
        self.n_choice[arm] += 1
        self.sample_mean[arm] = update_mean_inc(
            self.sample_mean[arm], reward, self.n_choice[arm], update_factor=self.mean_update_factor
        )

    def play(self):
        chosen_arm = self.select_arm()
        return chosen_arm
