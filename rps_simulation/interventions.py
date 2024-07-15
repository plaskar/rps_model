import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from rps_simulation.learning_curves import exponential_learning, logistic_learning
from rps_simulation.forgetting_curves import exponential_forgetting, power_forgetting 
from rps_simulation.practice_rate import simple_linear_rate, tmt_hyperbolic_rate
from rps_simulation.waiting_times import exponential_waiting_time 
from rps_simulation.rps_base import RPS_Basic, RPS_Basic_Multirun



##############################################################################
#    1. Run model with intervention on 'a'
##############################################################################

