import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from rps_simulation.learning_curves import exponential_learning
from rps_simulation.forgetting_curves import exponential_forgetting 
from rps_simulation.practice_rate import simple_linear_rate, tmt_hyperbolic_rate
from rps_simulation.waiting_times import exponential_waiting_time 




##############################################################################
#    1.  RPS Quittime - Run simulation till quit and then see career length
##############################################################################

class RPS_quittime: 

    def __init__(self, 
                 learning_func = exponential_learning(), # by default, we have exponential update s_new = s_old + alpha*(1-s_old) 
                 forgetting_func = exponential_forgetting(), # default is exponential forgetting
                 practice_rate_func = simple_linear_rate(), # default is simple_linear_rate 
                 waiting_time_dist = exponential_waiting_time, # default is exponential (NOT Pareto) waiting times 

                 ## Initial conditions and time-range:
                 initial_skill=0.1, initial_practice_rate=1, quit_thresh=20, max_time=1000):
        
        ## Simulation Attributes
        self.waiting_time_dist = waiting_time_dist
        self.learning_func = learning_func
        self.forgetting_func = forgetting_func
        self.practice_rate_func = practice_rate_func
        
        # ------- Data ------------
        # starting values and time_window
        self.initial_skill = initial_skill
        self.initial_practice_rate = initial_practice_rate
        self.quit_thresh = quit_thresh # if waiting time is higher than this, we assume learner has hard-quit
        self.max_time = max_time # even if quit_thresh is not met, we quit after max_time
        
        # Initialize empty lists for simulation results
        self.practice_times = [] # looks like [0, t1, t2...tn, max_time]
        self.skill_levels = [] # looks like [initial_skill, s1, ..., sn, s_final]
        self.practice_rates = [] # looks like [ lambda_0, lambda_1,....lambda_n, lambda_final]
        self.time_lags = [] # list of time_lags between practice events. Length = (# of practice_events) - 1
        self.forgetting_rates = [] # list of forgetting_rates after each practice event. Useful if 'spacing' is not None
            
        # Summary attributes: 
        self.final_skill = None # final skill at t = max_time, the end of the time-window
        self.final_practice_rate = None # final_practice_rate = practice_rate
        self.total_practice_events = None # total practice events during the run
        self.total_learn_time = None
        self.career_length = None



    #------- 1.1 Method to run simulation -------#
    def run_simulation(self):
        """
        Runs one instance of the simulation
        This generates the run data which includes practice_times, skill_levels, practice_rates, time_lags, etc.
        These are all class attributes initialized above. 
        """
        self.practice_times = [0]
        self.skill_levels = [self.initial_skill]
        self.practice_rates = [self.initial_practice_rate]
        self.time_lags = [] # only filled when 2 or more practice-events (PEs) have occured
        # self.forgetting_rates = []  # only filled when 1 or more PEs have occured
        
        flag = 0 # Simulation stops when flag = 1
        while flag==0: 
            current_time = self.practice_times[-1]
            current_skill = self.skill_levels[-1]
            current_practice_rate = self.practice_rates[-1]
            
            # Calculate time until next practice event
            wait_time = self.waiting_time_dist(current_practice_rate)
            next_prac_time = current_time + wait_time
            
            
            # Check for quit condition: 
            if wait_time > self.quit_thresh or current_skill < 0.01:
                flag = 1
                self.quit = 1
                self.career_length = current_time # agent waits too long, so quit
            
            if next_prac_time > self.max_time:
                flag = 1
                self.quit = 0
                self.career_length = self.max_time # agent practices till max_time

            # If flag == 1, calculate final values and break
            if flag == 1:
                final_skill = self.forgetting_func.calculate(current_skill, wait_time)
                final_practice_rate = self.practice_rate_func.calculate(self.skill_levels)  # same final_practice rate as at the last practice_event
                self.practice_times.append(next_prac_time)
                self.skill_levels.append(final_skill) 
                self.final_skill = final_skill 
                self.practice_rates.append(final_practice_rate)
                # self.career_length = self.max_time # player still kept practicing
                break

            
            # Calculate skill level just before next practice event
            skill_before_prac = self.forgetting_func.calculate(current_skill, wait_time)
            
            # Calculate skill level just after practice event
            skill_after_prac = self.learning_func.updated_skill(skill_before_prac)
            
            # Calculate practice rate for next practice event
            next_practice_rate = self.practice_rate_func.calculate(skill_history = [self.skill_levels, skill_after_prac])
            
            
            # Add skill, prac-event time and prac-rate to data
            self.skill_levels.append(skill_after_prac) 
            self.practice_times.append(next_prac_time)
            self.practice_rates.append(next_practice_rate)

            # Fill up time_lags and forgetting-rates list:
            if len(self.practice_times) >= 3: # at least 2 practice-events have occured:
                self.time_lags.append(self.practice_times[-1] - self.practice_times[-2])


            
        # Simulation summary data: 
        self.final_skill = self.skill_levels[-1]
        self.final_practice_rate = self.practice_rates[-1]
        self.total_practice_events = len(self.practice_times) - 2

        return self

