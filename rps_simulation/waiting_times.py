import numpy as np

def exponential_waiting_time(practice_rate):
    dt = np.random.exponential(scale=1/practice_rate)
    return dt
    
def pareto_waiting_time(practice_rate, min_wait = 0.01): 
    """
    min_wait = minimum waiting time, which is needs to be given to draw from pareto distributions
    """
    dt = (np.random.pareto(practice_rate) + 1)*min_wait   # waiting time dt
    return dt