from EpsilonGreedy import EpsilonGreedy

# First experiment
'''models = {
    'eps-greedy(0.0)': EpsilonGreedy(n_arms, 0.0),
    'eps-greedy(0.01)': EpsilonGreedy(n_arms, 0.01),
    'eps-greedy(0.1)': EpsilonGreedy(n_arms, 0.1)
    }'''

# Second experiment
models = {
    'eps-greedy(0.0)': EpsilonGreedy(n_arms, 0.0, optimistic_factor=5.0),
    'eps-greedy(0.1)': EpsilonGreedy(n_arms, 0.1)
    }
