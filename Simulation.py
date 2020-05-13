import numpy as np

class Simulation():
    def __init__(self, models, n_arms, n_timestpes):
        self.n_arms = n_arms
        self.n_timestpes = n_timestpes
        self.n_models = len(models)
        self.models = models
        self.opt_arm_ratio = np.zeros((self.n_models, self.n_timestpes))
        self.regret = np.zeros((self.n_models, self.n_timestpes))
        self.cumulative_reward = np.zeros((self.n_models, self.n_timestpes))

    def reset(self):
        del(self.opt_arm_ratio, self.regret, self.cumulative_reward)
        self.opt_arm_ratio = np.zeros((self.n_models, self.n_timestpes))
        self.regret = np.zeros((self.n_models, self.n_timestpes))
        self.cumulative_reward = np.zeros((self.n_models, self.n_timestpes))

    def set_problem(self, means, stds, opt_arm):
        self.means = means
        self.stds = stds
        self.opt_arm = opt_arm
        self.opt_reward = max(means)

    def generate_reward(self, arm):
        return np.random.normal(self.means[arm], self.stds[arm])
       
    def run(self):
        for t in np.arange(self.n_timestpes):
            for idx, (model_name, model) in enumerate(self.models.items()):  
                if str.find(model_name, 'UCB') != -1:
                    arm = model.play(t)
                else:    
                    arm = model.play()
                if self.opt_arm == arm:
                    self.opt_arm_ratio[idx][t] += 1
                reward = self.generate_reward(arm)
                model.update(arm, reward)
                self.regret[idx][t] += (self.opt_reward - reward) 
                self.cumulative_reward[idx][t] += reward 

        return self.opt_arm_ratio, self.regret, self.cumulative_reward



    