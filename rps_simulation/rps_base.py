import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from rps_simulation.learning_curves import exponential_learning
from rps_simulation.forgetting_curves import exponential_forgetting 
from rps_simulation.practice_rate import simple_linear_rate, tmt_hyperbolic_rate
from rps_simulation.waiting_times import exponential_waiting_time 


##############################################################################
#    1. RPS_core Class runs one instant of the RPS model,
#       giving the learning trajectory
##############################################################################

class RPS_core:
    """
    The core RPS Model Class makes one run of the model.
    * It has a constant forgetting and learning rate by default. Optionally, 
        spacing effect can be added to use variable forgetting rates. 
    
    * Users have to specify:
        1. "waiting_time_dist": The waiting time distribution
        2. "learning_func": This should tell us how to update the skill, based on current skill
        3. "forgetting_func": The forgetting function - exponential, power or something else. 
            It will use the forgetting_rate which is fixed
        4. "practice_rate_func": This should provide a positive practice rate (which controls wiating time)

    * Users may -Optionally- specify:
        1. "deadline_dict": A dictionary specifyig timings of deadlines, their weights (subjective importance)
            and a temporal-motivation-theory (tmt) effect function. See __init__ for more detials.
            
        2. "spacing_func": The functional form of the spacing effect. It will update forgetting_rate after each
            practive event (except the first practice-event), based on time-lags and hyperparameters. 
            Default is None. 
                    
    * The practice function takes as input the skill_levels history so far and generates an output practice rate
      For the basic model, this is simply rate = a + b*skill_level[-1]
    
    * Multiple runs of the simulation for a fixed learning curve will be done in the 'RPS_Mulirun' class
    """
    
    def __init__(self, 
                 learning_func = exponential_learning(), # by default, we have exponential update s_new = s_old + alpha*(1-s_old) 
                 forgetting_func = exponential_forgetting(), # default is exponential forgetting
                 practice_rate_func = simple_linear_rate(), # default is simple_linear_rate 
                 waiting_time_dist = exponential_waiting_time, # default is exponential (NOT Pareto) waiting times 
                 
                 ## Optionally add dictionary with deadlines, weights and tmt_effect function:
                 deadline_dict = {'deadlines': None, # if None then no deadlines, else add list of deadlines timings e.g. [33, 67, 100]
                                  'deadline_weights': None,  # If deadlines is not None, you can optionally add list of their weights,
                                          # e.g. [10, 10, 20] for 2 quizzes and a more important end-sem exam.
                                  'tmt_effect': None # assign tmt_effect class, e.g. tmt_hyperbolic_rate
                                 },
                 
                 ## Optionally set the spacing function (Default is no spacing) 
                 spacing_func = None,  
                 
                 ## Initial conditions and time-range:
                 initial_skill=0.1, initial_practice_rate=1, max_time=100):
        
        ## parameters of the RPS_core class:
        self.waiting_time_dist = waiting_time_dist
        self.learning_func = learning_func
        self.forgetting_func = forgetting_func
        self.practice_rate_func = practice_rate_func

        # Deadlines (optional):
        self.deadlines = deadline_dict['deadlines']
        self.deadline_weights = deadline_dict['deadline_weights']
        self.tmt_effect = deadline_dict['tmt_effect']
        

        # Spacing Effect (optional):
        self.spacing_func = spacing_func # By default = None 
        if spacing_func is not None:
            # set beta_max = starting forgetting rate
            self.spacing_func.set_beta_max(self.forgetting_func.forgetting_rate) 
        
        # ------- Data ------------
        # starting values and time_window
        self.initial_skill = initial_skill
        self.initial_practice_rate = initial_practice_rate
        self.max_time = max_time
        
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
        
                            
        
    #------- 1.1 Method to run simulation -------#
    def run_simulation(self):
        """
        Runs one instance of the RPS simulation (optionally with deadlines, spacing effect).
        This generates the run data which includes practice_times, skill_levels, practice_rates, time_lags, etc.
        These are all class attributes initialized above. 
        """
        self.practice_times = [0]
        self.skill_levels = [self.initial_skill]
        self.practice_rates = [self.initial_practice_rate]
        self.time_lags = [] # only filled when 2 or more practice-events (PEs) have occured
        self.forgetting_rates = []  # only filled when 1 or more PEs have occured
        
        
        while self.practice_times[-1] < self.max_time:
            current_time = self.practice_times[-1]
            current_skill = self.skill_levels[-1]
            current_practice_rate = self.practice_rates[-1]
            
            # Calculate time until next practice event
            wait_time = self.waiting_time_dist(current_practice_rate)
            next_prac_time = current_time + wait_time

            # If next practice time is beyond max_time, calculate final skill level:
            if next_prac_time > self.max_time:
                final_skill = self.forgetting_func.calculate(current_skill, self.max_time - current_time)
                final_practice_rate = self.practice_rate_func.calculate(self.skill_levels)  # same final_practice rate as at the last practice_event
                self.practice_times.append(self.max_time)
                self.skill_levels.append(final_skill) 
                self.final_skill = final_skill 
                self.practice_rates.append(final_practice_rate)
                #self.final_practice_rate = final_practice_rate
                break
            
            # Calculate skill level just before next practice event
            skill_before_prac = self.forgetting_func.calculate(current_skill, wait_time)
            
            # Calculate skill level just after practice event
            skill_after_prac = self.learning_func.updated_skill(skill_before_prac)
            
            # Calculate practice rate for next practice event
            next_practice_rate = self.practice_rate_func.calculate(skill_history = [self.skill_levels, skill_after_prac])
            
            # add deadline-effect to update practice-rate (optinal) if not None
            if self.deadlines is not None: 
                deadline_effect = self.tmt_effect.calculate(self.deadlines, self.deadline_weights, current_time, [self.skill_levels, skill_after_prac])
                next_practice_rate += deadline_effect # adding effect of deadline
            
            # Add skill, prac-event time and prac-rate to data
            self.skill_levels.append(skill_after_prac) 
            self.practice_times.append(next_prac_time)
            self.practice_rates.append(next_practice_rate)
            
            # Fill up time_lags and forgetting-rates list:
            if len(self.practice_times) >= 3: # at least 2 practice-events have occured:
                self.time_lags.append(self.practice_times[-1] - self.practice_times[-2])

            # Fill up forgetting_rates lists:
            if self.spacing_func is None: # no spacing
                self.forgetting_rates.append(self.forgetting_func.forgetting_rate) # forgetting rate stays constant throughout
            else: # spacing_func is not None
                if len(self.practice_times)==2: # first PE just occured:
                    self.forgetting_rates.append(self.forgetting_func.forgetting_rate)
                else: # more than 1 PE
                    next_forgetting_rate = self.spacing_func.calc_forgetting_rate(wait_times=self.time_lags)
                    self.forgetting_rates.append(next_forgetting_rate)
        
        
        # Simulation summary data: 
        self.final_skill = self.skill_levels[-1]
        self.final_practice_rate = self.practice_rates[-1]
        self.total_practice_events = len(self.practice_times) - 2

        return self

    #------ 1.2 method to get a dictionary of summary attributes from a sim. ------
    def data(self): 
        """
        Constructing summary attributes dictionary to return.
        May be useful in some context, if not will remove later.
        """
        summary_attributes = {'final_skill': self.final_skill,
                              'final_practice_rate': self.final_practice_rate,
                              'total_practice_events': self.total_practice_events,
                              'time_lags': self.time_lags,
                              'forgetting_rates': self.forgetting_rates,
                              'practice_times': self.practice_times,
                              'skill_levels': self.skill_levels,
                              'practice_rates': self.practice_rates
                             }
        return summary_attributes

    
    # ----- 1.3 -----
    # The function generates data to make cute plots - this includes the forgetting phase of the learning curves.
    # The generate_run_data function does not do this and only returns the list of practice_times and corresponding skill_levels and practice_rates.  
    def interpolate_learning_trajectory_dynamic(self, least_count=0.01, min_points=10):
        """
        Interpolate the learning trajectory with dynamic density based on the least count of time increments.
    
        Parameters:
        - practice_times: List of practice times.
        - skill_levels: List of skill levels at each practice time.
        - forgetting_rate: Rate of forgetting.
        - least_count: The least count of time increments for interpolation.
        - n_points: Minimum Number of points to add between practice times if the gap is less than the least count.
    
        Returns:
        - Tuple of interpolated_practice_times and interpolated_skill_levels with dynamically added points.
        """

        # Local variables for interpolated values
        int_practice_times = []
        int_skill_levels = []

        # Ensure there's at least two points to interpolate between
        if len(self.practice_times) < 2:
            return self.practice_times, self.skill_levels

        for i in range(len(self.practice_times) - 1):
            start_time, end_time = self.practice_times[i], self.practice_times[i + 1]
            time_gap = end_time - start_time

        # Determine how many points to interpolate based on time_gap and least_count
            if time_gap <= least_count:
                # If the time gap is less than or equal to the least count, use fixed n_points
                times = np.linspace(start_time, end_time, min_points, endpoint=False)
            else:
                # Otherwise, divide the gap by the least_count to determine the number of points. At least min_points are added
                points_count = max(int(time_gap / least_count), min_points)  # Ensure at least min_points interpolated points
                times = np.linspace(start_time, end_time, points_count, endpoint=False)
            
            for t in times[:-1]:  # Exclude the last point to prevent overlap
                elapsed_time = t - start_time
                #interpolated_skill = self.skill_levels[i] * np.exp(-self.forgetting_rate * elapsed_time)
                interpolated_skill = self.forgetting_func.calculate(self.skill_levels[i], elapsed_time)
                int_practice_times.append(t)
                int_skill_levels.append(interpolated_skill)
        
        # Add the last original points
        int_practice_times.append(self.practice_times[-1])
        int_skill_levels.append(self.skill_levels[-1])

        return int_practice_times, int_skill_levels
    
    # ----- 1.4 Plot smoothed learning trajectory (with interpolated forgetting) -----
    def plot_learning_trajectory(self, least_count=0.01, min_points=10, overlay=None, col_parms=None, save_location=None, save_dpi=512):
        """Plots a smoothed learning trajectory including the forgetting curves interpolated  between practice-events"""

        color_dict = {'base':'#FF6B6B', 'obs_line':'black'} # color dict, can change using col_parms argument
        if col_parms is not None:
            color_dict.update(col_parms)
        
        interpolated_practice_times, interpolated_skill_levels = self.interpolate_learning_trajectory_dynamic(least_count, min_points)
        
        plt.figure(figsize=(10, 6))
        plt.plot(interpolated_practice_times, interpolated_skill_levels, linestyle='-', color=color_dict['base'])

        # Using 'overlay' parameter you can have either:
        #    1. Markers of the updated skill values at practice points = observed skill levels 
        #    2. An additional line plot of the smooth learning trajectory with forgetting part interpolated.
        #       Mind you, it makes the plot file size much larger. 
        # You can also change color using the 'col_parms' parameter.     
        if overlay == 'markers':
            plt.scatter(self.practice_times, self.skill_levels, marker='o', linestyle='-', color=color_dict['base'])
        elif overlay=='observed_line':
            plt.plot(self.practice_times, self.skill_levels, marker='o', linestyle='-', color=color_dict['obs_line'])
            
        plt.title('Learning Trajectory with Forgetting', fontsize=22)
        plt.xlabel('Time', fontsize=18)
        plt.ylabel('Skill', fontsize=18)
        plt.ylim([0,1]) # fix range of y-axis to 0-1
        plt.xlim([0, self.max_time]) # x-rane between 0 to max_time
        plt.grid(True, alpha=1)
        
        if save_location is not None:
            plt.savefig(save_location, dpi=save_dpi)
        plt.show()


    # ------ 1.5 Plot the counting process associated with the temporal practice point process -----  
    def practice_event_plot(self, color='black', lw=2):
        """Plot of the counting process associated with practice-events"""
        counts = [ i for i in range(1, len(self.practice_times)-1)] # y-axis values
        
        plt.figure(figsize=(10, 2))  # Wide and not too high
        plt.step(x=self.practice_times[1:-1], y=counts, color=color, linewidth=lw)
        plt.title('Practice Times', fontsize=22)
        plt.xlabel('Time', fontsize=18)
        #plt.yticks([])  # Hide y-axis ticks
        plt.tight_layout()
        plt.show()

    
    # ------- 1.6 Method to run Interventions during Simulation --------
    def run_sim_with_intervention(self, set_intervention_param=None):
        """
        Runs one instance of the RPS simulation along with intervention on a, b as specified
        by a dictionary. 
        """

        # Default intervention parameters. User can change some/all of them through setting intervention_param
        int_param={'S_min_thresh': 0.05, # min skill below which intervention kicks in
                   'time_min_thresh': 20, # min. time before starting intervention, giving learners chance to learn on their own.  
                   'a_high':2, # high value to which a is set during intervention
                   'b_high': self.practice_rate_func.b, # high value to which b is set during intervention
                   'time_int': 10, # min. time for which intervention is done. Intervention ends on the next practice point  
                   'a_low': self.practice_rate_func.a, # value to which a is set after intervention
                   'b_low': self.practice_rate_func.b # value to which b is set after intervention
                    }
        
        # If user has provided custom params, update:
        if set_intervention_param is not None:
            int_param.update(set_intervention_param)

        # Initializing Sim Data Lists:
        self.practice_times = [0]
        self.skill_levels = [self.initial_skill]
        self.practice_rates = [self.initial_practice_rate]
        self.time_lags = [] # only filled when 2 or more practice-events (PEs) have occured
        self.forgetting_rates = []  # only filled when 1 or more PEs have occured
        
        self.int_status = [0] # list filled with 1/0 to see status of intervention at each practice-event
        self.a_vals = [self.practice_rate_func.a] # list filled with a values, same length as practice_times
        self.b_vals = [self.practice_rate_func.b] # list filled with b values, same length as practice_times
        
        curr_int_status = 0 # turns 1 if intervention is on
        curr_int_time = 0 #  running count of how long intervention has been applied
        
        while self.practice_times[-1] < self.max_time:
            current_time = self.practice_times[-1]
            current_skill = self.skill_levels[-1]
            current_practice_rate = self.practice_rates[-1]

            # Calculate time until next practice event
            wait_time = self.waiting_time_dist(current_practice_rate)
            next_prac_time = current_time + wait_time

            # add to intervention time if intervention is on
            if curr_int_status == 1:
                curr_int_time += wait_time
            

            # If next practice time is beyond max_time, calculate final skill level:
            if next_prac_time > self.max_time:
                final_skill = self.forgetting_func.calculate(current_skill, self.max_time - current_time)
                final_practice_rate = self.practice_rate_func.calculate(self.skill_levels)  # same final_practice rate as at the last practice_event
                
                self.practice_times.append(self.max_time)
                self.skill_levels.append(final_skill) 
                self.practice_rates.append(final_practice_rate)
                self.int_status.append(curr_int_status)
                self.a_vals.append(self.practice_rate_func.a)
                self.b_vals.append(self.practice_rate_func.b)
                break
            
            # Calculate skill level just before next practice event
            skill_before_prac = self.forgetting_func.calculate(current_skill, wait_time)
            
            # Calculate skill level just after practice event
            skill_after_prac = self.learning_func.updated_skill(skill_before_prac)

            # Chcek for intervention condition and apply intervention as applicable:
            if skill_after_prac < int_param['S_min_thresh'] and next_prac_time>= int_param['time_min_thresh'] and curr_int_time <= int_param['time_int']:
                curr_int_status = 1 
                self.practice_rate_func.a = int_param['a_high']
                self.practice_rate_func.b = int_param['b_high']
            # if intervention has lasted till time_int, set it off:
            elif curr_int_time > int_param['time_int']: 
                curr_int_status = 0
                self.practice_rate_func.a = int_param['a_low']
                self.practice_rate_func.b = int_param['b_low']
            
            # Calculate practice rate for next practice event
            next_practice_rate = self.practice_rate_func.calculate(skill_history = [self.skill_levels, skill_after_prac])
            
            # add deadline-effect to update practice-rate (optinal) if not None
            if self.deadlines is not None: 
                deadline_effect = self.tmt_effect.calculate(self.deadlines, self.deadline_weights, current_time, [self.skill_levels, skill_after_prac])
                next_practice_rate += deadline_effect # adding effect of deadline
            
            # Add skill, prac-event time, prac-rate, a-val and b-val to data
            self.skill_levels.append(skill_after_prac) 
            self.practice_times.append(next_prac_time)
            self.practice_rates.append(next_practice_rate)
            self.int_status.append(curr_int_status)
            self.a_vals.append(self.practice_rate_func.a)
            self.b_vals.append(self.practice_rate_func.b)
            
            # Fill up time_lags and forgetting-rates list:
            if len(self.practice_times) >= 3: # at least 2 practice-events have occured:
                self.time_lags.append(self.practice_times[-1] - self.practice_times[-2])

            # Fill up forgetting_rates lists:
            if self.spacing_func is None: # no spacing
                self.forgetting_rates.append(self.forgetting_func.forgetting_rate) # forgetting rate stays constant throughout
            else: # spacing_func is not None
                if len(self.practice_times)==2: # first PE just occured:
                    self.forgetting_rates.append(self.forgetting_func.forgetting_rate)
                else: # more than 1 PE
                    next_forgetting_rate = self.spacing_func.calc_forgetting_rate(wait_times=self.time_lags)
                    self.forgetting_rates.append(next_forgetting_rate)
        
        # Simulation summary data: 
        self.final_skill = self.skill_levels[-1]
        self.final_practice_rate = self.practice_rates[-1]
        self.total_practice_events = len(self.practice_times) - 2

        return self



    
    # -----------------------------------------------------
    # --------- PLOTTING METHODS --------------------------
    # -----------------------------------------------------
    
    #------ 1.7 Plot simple line-plot of updated skill at practice events (no forgetting) ------
    def plot_simple_trajectory(self, save_location=None, save_dpi=512):
        """
        plot simple learning trajectory without the smoothed forgetting curve. 
        Skill at consecutive practice-events are joined by straight lines
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.practice_times, self.skill_levels, marker='o', linestyle='-', color='#FF6B6B')
        plt.title('Simple Learning Trajectory', fontsize=22)
        plt.xlabel('Time', fontsize=19)
        plt.ylabel('Skill', fontsize=19)
        plt.ylim([0,1]) # fix range of y-axis to 0-1
        plt.grid(True, alpha=1)
        if save_location is not None:
            plt.savefig(save_location, dpi=save_dpi)
        plt.show()

    #------ 1.8 Plot a rectangle of practice-event times ------
    def practice_times_plot(self, color='black', lw=1, save_location=None, save_dpi=512):
        """Plot practice_times as vertical lines to visually show when practice events occur"""
        plt.figure(figsize=(10, 2))  # Wide and not too high
        for prac_time in self.practice_times[1:-1]:  # Excluding the first and last elements
            plt.axvline(x=prac_time, color=color, linestyle='-', linewidth=lw)
        plt.title('Practice Times', fontsize=22)
        plt.xlabel('Time', fontsize=18)
        plt.yticks([])  # Hide y-axis ticks
        plt.tight_layout()
        if save_location is not None:
            plt.savefig(save_location, dpi=save_dpi)
        plt.show()

    # ----- 1.9 Plot skill trajectory with intervention information ----
    def plot_intervention_trajectory(self, save_location = None, save_dpi=512):
        """
        plot simple trajectory with shaded intervention region without the smoothed forgetting curve. 
        Skill at consecutive practice-events are joined by straight lines
        """
        int_t_start =  None
        int_t_stop =  None
        int_status = self.int_status
        
        # Loop through the list and look for transitions
        for i in range(1, len(int_status)):
            if int_status[i-1] == 0 and int_status[i] == 1:
                if int_t_start is None:  # Ensure idx1 is set only the first time the transition occurs
                    int_t_start = i
            elif int_status[i-1] == 1 and int_status[i] == 0:
                int_t_stop = i
                break  
            
        plt.figure(figsize=(10, 6))
        plt.plot(self.practice_times, self.skill_levels, marker='o', linestyle='-', color='#FF6B6B')
        # add intervention shaded region:
        plt.axvspan(self.practice_times[int_t_start], self.practice_times[int_t_stop], color='grey', alpha=0.5, label='Intervention On')
        plt.title('Learning trajectory with intervention', fontsize=22)
        plt.legend(fontsize=20)
        plt.xlabel('Practice Time', fontsize=19)
        plt.ylabel('Skill Level', fontsize=19)
        plt.grid(True, alpha=1)
        if save_location is not None:
            plt.savefig(save_location, dpi=save_dpi)
        plt.show()

    
    # ----- 1.10 PLotting simple trajectory + event time info ----
    def plot_combined_trajectory(self, colour= '#FF6B6B', save_location=None, save_dpi=512):
        """
        Plot simple learning trajectory and practice times in a single figure.
        """
        # set subfigures:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), 
                                   gridspec_kw={'height_ratios': [6, 1]},
                                   sharex=True)
        
        # Create a larger subplot for the learning trajectory
        ax1.plot(self.practice_times, self.skill_levels, marker='o', linestyle='-', color=colour)
        ax1.set_title('Simple Learning Trajectory', fontsize=22)
        ax1.set_ylabel('Skill', fontsize=19)
        ax1.set_ylim([0, 1])
        ax1.grid(True, alpha=1)
        
        # Remove x-axis labels from the top plot
        #ax1.set_xticklabels([])
        
        # Create a smaller subplot for practice times
        #ax2 = plt.subplot2grid((4, 1), (3, 0), rowspan=1, sharex=ax1)
        for prac_time in self.practice_times[1:-1]:  # Excluding the first and last elements
            ax2.axvline(x=prac_time, color='black', linestyle='-', linewidth=1)
        
        #ax2.set_title('Practice Events', fontsize=22)
        ax2.set_xlabel('Time', fontsize=19)
        ax2.set_ylabel('Practice\nEvents', fontsize=14)
        ax2.set_yticks([])  # Hide y-axis ticks
        
        # Adjust x-axis ticks if needed
        ax2.set_xticks([int(x) for x in np.linspace(0, self.max_time, 11)])
        
        # Adjust the layout and spacing
        plt.tight_layout()
        
        if save_location is not None:
            plt.savefig(save_location, dpi=save_dpi)
        plt.show()
    


