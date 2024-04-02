import numpy as np

def exponential_waiting_time(practice_rate):
    return np.random.exponential(scale=1/practice_rate)
