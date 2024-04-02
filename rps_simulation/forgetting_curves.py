import numpy as np 

def exponential_forgetting(skill, time, forgetting_rate):
    return skill * np.exp(-forgetting_rate * time)

