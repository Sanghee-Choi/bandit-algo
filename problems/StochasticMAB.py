from problems.MAB import BanditNormalProblem, BanditUniformProblem
import numpy as np

class BernoulliBanditProblem(BanditUniformProblem):
    # bandit rewards: list of 
    def set_problem(self, bandit_params=None):
        if bandit_params is None:
            # if there is no bandit params, then a problem is generated.
            # - the means for arms are drawn from a uniform distribution with an interval [0, 1]
            self.bandit_rewards = self.problem_generator()
        else:
            try:
                self.bandit_rewards = bandit_params['bandit_rewards']
            except KeyError:
                print('To set a problem, the `bandit_params` dict as an argument should')
                print('> have a key `bandit_rewards` that stores expected rewards.')
                exit(0)   
        self.optimal_arm = np.argmax(self.bandit_rewards)
        return 

    def gen_reward(self, arm, t):
        sample = np.random.binomial(1, self.bandit_rewards[arm])
        instantaneous_regret = self.bandit_rewards[self.optimal_arm] - self.bandit_rewards[arm]
        is_opt = self.optimal_arm == arm
        return sample, instantaneous_regret, is_opt 


class NormalBanditProblem(BanditNormalProblem):
    def set_problem(self, bandit_params=None):
        if bandit_params is None:
            # if there is no bandit params, then a problem is generated.
            # - the means for arms are drawn from a normal distribution with mean 0 and variance 1
            # - the standard deviation for all the arms is 1.
            self.bandit_rewards, self.bandit_stds = self.problem_generator()
        else:
            try:
                self.bandit_rewards = bandit_params['bandit_rewards']
                self.bandit_stds = bandit_params['bandit_stds']
            except KeyError:
                print('To set a problem, the `bandit_params` dict as an argument should ')
                print('> have a key `bandit_rewards` whose value is expected rewards,')
                print('> and have a key `bandit_stds` whose value is standard deviations.')
                exit(0)  
        self.optimal_arm = np.argmax(self.bandit_rewards)
        return 

    def gen_reward(self, arm, t):
        sample = np.random.normal(self.bandit_rewards[arm], self.bandit_stds[arm])
        instantaneous_regret = self.bandit_rewards[self.optimal_arm] - self.bandit_rewards[arm]
        is_opt = self.optimal_arm == arm
        return sample, instantaneous_regret, is_opt 

