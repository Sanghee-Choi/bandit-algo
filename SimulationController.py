from Simulation import Simulation
from algorithms.EpsilonGreedy import EpsilonGreedy
from algorithms.UCB import UCB
from problems.StochasticMAB import NormalBanditProblem

import numpy as np

class SimulationController():
    def __init__(self, n_arms, n_runs, n_timesteps, models, problem=None, seed=100):
        self.n_arms = n_arms
        self.n_runs = n_runs

        # model is passed in the list format whose element is an object of a bandit model.
        self.models = models
        self.n_models = len(self.models)

        # problem
        self.problem = NormalBanditProblem(n_arms) if problem is None else problem

        # object for simulation of each run
        self.simulation = Simulation(self.models, self.n_arms, self.problem, n_timesteps)

        # storage for performance metrics
        self.opt_arm_ratio = np.zeros((self.n_models, n_timesteps))
        self.regret = np.zeros((self.n_models, n_timesteps))
        self.cumulative_reward = np.zeros((self.n_models, n_timesteps))
        np.random.seed(seed)

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
        for _ in np.arange(self.n_runs):
            self.problem.set_problem()
            opt_arm_ratio_for_run, regret_for_run, cumulative_reward_for_run = self.simulation.run()
            #print(opt_arm_ratio_for_run)
            self.update_results(opt_arm_ratio_for_run, regret_for_run, cumulative_reward_for_run)
            self.reset()
        #print(self.opt_arm_ratio)
        self.finalize_results()
        return self.opt_arm_ratio, self.regret, self.cumulative_reward

