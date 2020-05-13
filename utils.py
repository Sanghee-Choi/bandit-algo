def update_mean_inc(prev_mean, sample_reward, n):
    return prev_mean + (1.0 / n * (sample_reward - prev_mean))

def update_var_inc(prev_var, prev_mean, cur_mean, sample_reward, n):
    if n > 0:
        return prev_var + (sample_reward - prev_mean) * (sample_mean - cur_mean)
    else:
        return 