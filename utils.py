import matplotlib.pyplot as plt

def update_mean_inc(prev_mean, sample_reward, n, update_factor=None):
    '''
        incremental average
        - update_factor: determining how to update the mean value
        - if it is 1.0/n, this function calculate the mean.
        - update_factor can be set by the `update_factor` parameter.
    '''
    update_factor = 1.0/n if update_factor is None else update_factor
    return prev_mean + (update_factor * (sample_reward - prev_mean))

def update_var_inc(prev_var, prev_mean, cur_mean, sample_reward, n):
    if n > 0:
        return prev_var + (sample_reward - prev_mean) * (sample_reward - cur_mean)
    else:
        return 

def draw_plot(data, algos, ylabel, xlabel='time', loc='lower right', **kwargs):
    fig = plt.figure(figsize=(8, 7))
    for i in range(data.shape[0]):
        plt.plot(data[i], label=algos[i])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if 'ylim' in kwargs:
        plt.ylim(kwargs.get('ylim'))
    plt.legend(loc=loc)
    plt.show()