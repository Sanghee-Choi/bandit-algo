import numpy as np

class Simulation():
    def __init__(self, models, n_arms, problem, n_timestpes):
        self.n_arms = n_arms
        self.problem = problem
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
       
    def run(self):
        for t in np.arange(self.n_timestpes):
            for idx, (model_name, model) in enumerate(self.models.items()):  
                if str.find(model_name, 'UCB') != -1:
                    arm = model.play(t)
                else:    
                    arm = model.play()
                reward, instantaneous_regret, is_opt = self.problem.gen_reward(arm, t)
                model.update(arm, reward)
                self.regret[idx][t] = self.regret[idx][t-1] + instantaneous_regret if t != 0 else instantaneous_regret
                if is_opt == True:
                    self.opt_arm_ratio[idx][t] += 1
                self.cumulative_reward[idx][t] = self.cumulative_reward[idx][t-1] + reward if t != 0 else reward

        return self.opt_arm_ratio, self.regret, self.cumulative_reward



    