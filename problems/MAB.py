from abc import ABC, abstractmethod
import numpy as np

class BanditProblem(ABC):
    @abstractmethod
    def __init__(self, n_arms):
        pass 

    @abstractmethod
    def gen_reward(self, arm, t):
        pass 


class BanditNormalProblem(BanditProblem):
    def __init__(self, n_arms):
        self.n_arms = n_arms
        return

    def gen_reward(self, arm, t):
        return NotImplementedError
    
    def problem_generator(self):
        '''
            if there is no bandit params, then a problem is generated.
            - The means for arms are drawn from a normal distribution with mean 0 and variance 1
            - The standard deviation for all the arms is 1.
        '''
        self.means = np.random.normal(0, 1, self.n_arms)
        self.stds = np.ones(self.n_arms, dtype=float)
        return self.means, self.stds


class BanditUniformProblem(BanditProblem):
    def __init__(self, n_arms):
        self.n_arms = n_arms
        return

    def gen_reward(self, arm, t):
        return NotImplementedError
    
    def problem_generator(self):
        ''' 
            This is a default problem generator of this class
            - The means for arms are independently drawn from a uniform distribution with an interval [0, 1]
        '''
        self.means = np.random.uniform(size=self.n_arms)
        return self.means
