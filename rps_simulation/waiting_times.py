import numpy as np


# The standard exponential waiting in the basic model:
def exponential_waiting_time(practice_rate, min_wait = 0.2):
    dt = max(np.random.exponential(scale = 1/practice_rate), min_wait) # waiting time dt
    return dt
    


# Log-normal waiting time distribution:
def log_normal_waiting_time(practice_rate, sigma=1, min_wait=0.2):
    """
    log-normal distribution takes two values: mean and sigma
    mean is the mean of the underlying normal distribution
    sigma is the standard deviation of the underlying normal distribution

    we set mean = 1/practice_rate
    sigma can be another parameter to describe the learner's motivation state. 
    """
    mean = 1/practice_rate # mean of the underlying normal distribution

    dt = np.random.normal(mean, sigma) # waiting time dt

    return max(min_wait, dt) # waiting time cannot be negative




# the following leads to very weird simulation behaviour where agents keep switching between 0 and 1 in skill.
# problem 1: too many very long wait times where agents forget everything. 
# problem 2: too many very short wait times and a lot of practice, leading to skill of 1
def pareto_waiting_time(practice_rate, min_wait = 0.01): 
    """
    min_wait = minimum waiting time, which is needs to be given to draw from pareto distributions
     --- Read numpy documentation of random.pareto for more details ----
    """
    dt = (np.random.pareto(practice_rate) + 1)*min_wait   # waiting time dt

    return dt



