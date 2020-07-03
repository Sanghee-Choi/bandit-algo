from problems.MAB import BanditNormalProblem, BanditUniformProblem
import numpy as np

class NonstationaryBernoulliProblem(BanditUniformProblem):
    def __init__(self, n_arms, period, mode='dist'):
        '''
            n_arms: the number of arms
            period: the period by which the means are changing
                > if the mode is 'linear', then period is equal to the total timesteps
            mode:
                > dist: the mean of bernoulli distribution is drawn from the uniform distribution in [0, 1]
                > linear: the mean is changing linearly by two parameters, bandit_rewards1 and bandit rewards2
                    - bandit_rewards1 and bandit rewards2 can manually be set in the method `set_problem`
        '''
        self.n_arms = n_arms      
        self.period = period 
        self.mode = mode

    def set_problem(self, bandit_params=None, t=None):
        '''
            bandit_paras: dictionary to define the parameters of this problem class
            - keys
                - bandit_rewards1: used for 'dist' and 'linear' mode
                - bandit_rewards2: only used for 'linear' mode
            - If you want to set parameters manually, then each reward passed into must have `n_arms` elements.
        '''
        if self.mode == 'dist':
            if bandit_params != None:
                if 'bandit_rewards1' in bandit_params:
                    self.bandit_rewards = bandit_params['bandit_rewards1']
                    if len(self.bandit_rewards) != self.n_arms:
                        print('the number of arms in `bandit_rewards` is not equal to total number of arms.')
                        exit(0)
                    self.optimal_arm = np.argmax(self.bandit_rewards)
                else:
                    print('A key `bandit_reward1` should be defined in `bandit_params` dictionary.')
                    exit(0)
            else:
                self.bandit_rewards = self.problem_generator()
                self.optimal_arm = np.argmax(self.bandit_rewards)
        elif self.mode == 'linear':
            if (bandit_params != None):
                if ('bandit_rewards1' in bandit_params) & ('bandit_rewards2' in bandit_params):
                    self.bandit_rewards1 = bandit_params['bandit_rewards1']
                    self.bandit_rewards2 = bandit_params['bandit_rewards2']
                    if (len(self.bandit_rewards1) != self.n_arms) or (len(self.bandit_rewards2) != self.n_arms):
                        print('the number of arms in `bandit_rewards1` or `bandit_rewards2` is not equal to total number of arms.')
                        exit(0)
                    self.optimal_arm = np.argmax(self.bandit_rewards)
                else:
                    print('A key `bandit_rewards1` and `bandit_rewards2` should be defined in `bandit_params` dictionary.')
                    exit(0)
            else:
                self.bandit_rewards1 = self.problem_generator()
                self.bandit_rewards2 = self.problem_generator()
        return 

    def get_problem(self, t):
        '''
            Follow the mode to change the expected reward according to each timestep
        '''
        if self.mode == 'dist':
            if (t % self.period == 0) & (t != 0):
                self.bandit_rewards = self.problem_generator()
                self.optimal_arm = np.argmax(self.bandit_rewards)
        elif self.mode == 'linear':
            if (t <= self.period) & (t >= 0):
                self.bandit_rewards = ((1 - t/self.period)*self.bandit_rewards1) + (t/self.period*self.bandit_rewards2)
                self.optimal_arm = np.argmax(self.bandit_rewards)

    def gen_reward(self, arm, t):
        self.get_problem(t=t)
        sample = np.random.binomial(1, self.bandit_rewards[arm])
        instantaneous_regret = self.bandit_rewards[self.optimal_arm] - self.bandit_rewards[arm]
        is_opt = self.optimal_arm == arm
        return sample, instantaneous_regret, is_opt 
