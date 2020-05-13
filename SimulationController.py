from Simulation import Simulation
from EpsilonGreedy import EpsilonGreedy
from UCB import UCB

import numpy as np

class SimulationController():
    def __init__(self, n_arms, n_runs, n_timesteps, seed=100):
        self.n_arms = n_arms
        self.n_runs = n_runs
        self.models = models = {
            'UCB(2.0)': UCB(n_arms, 2.0),
            'UCB(1.0)': UCB(n_arms, 1.0),
            'eps-greedy(0.1)': EpsilonGreedy(n_arms, 0.1),
            'eps-greedy(0.05)': EpsilonGreedy(n_arms, 0.05)
            }
        self.n_models = len(self.models)
        self.simulation = Simulation(self.models, self.n_arms, n_timesteps)
        self.opt_arm_ratio = np.zeros((self.n_models, n_timesteps))
        self.regret = np.zeros((self.n_models, n_timesteps))
        self.cumulative_reward = np.zeros((self.n_models, n_timesteps))
        np.random.seed(seed)

    def problem_generator(self):
        means = np.random.normal(0, 1, self.n_arms)
        stds = np.ones(self.n_arms, dtype=float)
        opt_arm = np.argmax(means)
        return means, stds, opt_arm

    def update_results(self, opt_arm_ratio_for_run, regret_for_run, cumulative_reward_for_run):
        self.opt_arm_ratio = self.opt_arm_ratio + opt_arm_ratio_for_run
        self.regret = self.regret + regret_for_run
        self.cumulative_reward = self.cumulative_reward + cumulative_reward_for_run

    def finalize_results(self):
        self.opt_arm_ratio= self.opt_arm_ratio / self.n_runs
        self.regret = self.regret / self.n_runs
        self.cumulative_reward = self.cumulative_reward / self.n_runs

    def reset(self):
        for _, model in self.models.items():
            model.reset()
        self.simulation.reset()
    
    def run(self):
        for run in np.arange(self.n_runs):
            means, stds, opt_arm = self.problem_generator()
            self.simulation.set_problem(means, stds, opt_arm)
            opt_arm_ratio_for_run, regret_for_run, cumulative_reward_for_run = self.simulation.run()
            self.update_results(opt_arm_ratio_for_run, regret_for_run, cumulative_reward_for_run)
            self.reset()
        self.finalize_results()
        return self.opt_arm_ratio, self.regret, self.cumulative_reward

