import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from rps_simulation.learning_curves import exponential_learning
from rps_simulation.forgetting_curves import exponential_forgetting 
from rps_simulation.practice_rate import simple_linear_rate, tmt_hyperbolic_rate
from rps_simulation.waiting_times import exponential_waiting_time 


##############################################################################
#    1. RPS_Basic Class runs one instant of the basic RPS model,
#       giving the learning trajectory
##############################################################################

class RPS_Basic:
    """
    The Basic RPS Model Class makes one run of the model. poetry
    * It has a fixed forgetting rate, fixed learning rate. 
    
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
    
    * Multiple runs of the simulation for a fixed learning curve will be done in the next class
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
        
        ## parameters of the RPS_Basic class:
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
        
                            
        

    # Generates the run_data = which includes practice_times, skill_levels and practice_rates attributes of the class
    def run_simulation(self):
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
                self.practice_rates.append(final_practice_rate); self.final_practice_rate = final_practice_rate
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
        
        ## Filling up Simulation Data
        self.final_skill = self.skill_levels[-1]
        self.final_practice_rate = self.practice_rates[-1]
        self.total_practice_events = len(self.practice_times) - 2

        return self

    # Returns dictionary of data from the simulation-run
    def data(self): 
        """Constructing summary attributes dictionary to return"""
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

    
    def plot_simple_trajectory(self):
        """
        plot simple learning trajectory without the smoothed forgetting curve. 
        Skill at consecutive practice-events are joined by straight lines
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.practice_times, self.skill_levels, marker='o', linestyle='-', color='#FF6B6B')
        plt.title('Simple Learning Trajectory', fontsize=22)
        plt.xlabel('Practice Time', fontsize=19)
        plt.ylabel('Skill Level', fontsize=19)
        plt.grid(True, alpha=1)
        plt.show()
    
        
        
    
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
    
    
    def plot_learning_trajectory(self, least_count=0.01, min_points=10):
        """Plots a smoothed learning trajectory including the forgetting curves interpolated  between practice-events"""
        interpolated_practice_times, interpolated_skill_levels = self.interpolate_learning_trajectory_dynamic(least_count, min_points)
        
        plt.figure(figsize=(10, 6))
        # plt.plot(interpolated_practice_times, interpolated_skill_levels, marker='o', linestyle='-')
        plt.plot(interpolated_practice_times, interpolated_skill_levels, linestyle='-', color='#FF6B6B')
        plt.title('Learning Trajectory with Interpolation', fontsize=22)
        plt.xlabel('Practice Time', fontsize=18)
        plt.ylabel('Skill Level', fontsize=18)
        plt.grid(True, alpha=1)
        plt.show()
    

    def practice_times_plot(self, color='black', lw=1):
        """Plot practice_times as vertical lines to visually show when practice events occur"""
        plt.figure(figsize=(10, 2))  # Wide and not too high
        for prac_time in self.practice_times[1:-1]:  # Excluding the first and last elements
            plt.axvline(x=prac_time, color=color, linestyle='-', linewidth=lw)
        plt.title('Practice Times', fontsize=22)
        plt.xlabel('Time', fontsize=18)
        plt.yticks([])  # Hide y-axis ticks
        plt.tight_layout()
        plt.show()

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



##############################################################################
### 2. Class to have multiple runs of the basic model
##############################################################################

class RPS_Basic_Multirun:
    """
    Multiple Runs of the RPS_Basic class and store useful statistics about the simulation.
    Also allows plotting trajectories and final skill histograms, etc. to test how different
    learning and forgetting curves, deadlines, spacings etc. affect results. 
    This class is needed also to perform sensitivity analysis.
    """
    def __init__(self, waiting_time_dist, learning_func, forgetting_func, practice_rate_func, 
                 deadline_dict = {'deadlines': None, 'deadline_weights': None, 'tmt_effect': None},
                 spacing_func = None,
                 n_sims=1000, initial_skill=0.1, initial_practice_rate=1, max_time=100):
        
        # Class Attributes:
        self.waiting_time_dist = waiting_time_dist
        self.learning_func = learning_func
        self.forgetting_func = forgetting_func
        self.practice_rate_func = practice_rate_func

        # hyperparameters
        self.n_sims = n_sims  # Number of simulations to run
        self.initial_skill = initial_skill
        self.initial_practice_rate = initial_practice_rate
        self.max_time = max_time

        # Deadlines (optional):
        self.deadlines = deadline_dict['deadlines']
        self.deadline_weights = deadline_dict['deadline_weights']
        self.tmt_effect = deadline_dict['tmt_effect']

        # Spacing Effect (optional):
        self.spacing_func = spacing_func # By default = None 
        if spacing_func is not None:
            # set beta_max = starting forgetting rate
            self.spacing_func.set_beta_max(self.forgetting_func.forgetting_rate) 
        
        # Summary Data from all Runs:
        self.final_skills = []  # To store final skill levels from all sims
        self.total_practice_events = []  # To store the number of practice events from all sims
        
        self.all_skill_levels = [] # list of lists, contains skill_levels for each simulation run
        self.all_practice_times = [] # contains practice_time list for each simulation run
        self.all_practice_rates = [] # contains practice_time list for each simulation run
        self.all_time_lags = [] # contains time_lag list for each sim run
        self.all_forgetting_rates = [] # ditto for forgetting_rate for each sim run
        
        self.interpolated_skills = []
        self.interpolated_prac_times = []

    def run_multiple_sims(self):
        for _ in range(self.n_sims):
            model = RPS_Basic(waiting_time_dist=self.waiting_time_dist, learning_func=self.learning_func,
                              forgetting_func=self.forgetting_func, practice_rate_func=self.practice_rate_func,
                              deadline_dict = {'deadlines': self.deadlines, 'deadline_weights': self.deadline_weights, 'tmt_effect': self.tmt_effect},
                              spacing_func = self.spacing_func,
                              initial_skill=self.initial_skill, initial_practice_rate=self.initial_practice_rate, max_time=self.max_time)
            model.run_simulation() # run one instance of simulation
            
            # interpolating skills in-between practice events for smooth plots
            interpolated_practice_times, interpolated_skill_levels = model.interpolate_learning_trajectory_dynamic(least_count=0.01, min_points=10)

            # adding data from current sim
            self.final_skills.append(model.final_skill)
            self.total_practice_events.append(model.total_practice_events)
            self.all_skill_levels.append(model.skill_levels)
            self.all_practice_times.append(model.practice_times)
            self.all_time_lags.append(model.time_lags)
            self.all_forgetting_rates.append(model.forgetting_rates)
            
            # interpolated data:
            self.interpolated_prac_times.append(interpolated_practice_times)
            self.interpolated_skills.append(interpolated_skill_levels)
    
    def plot_final_skill_histogram(self, colour='blue', n_bins=50, save_location=False):
        plt.figure(figsize=(10, 6))
        plt.hist(self.final_skills, bins=[i/n_bins for i in range(n_bins+1)], color=colour, edgecolor='black')
        plt.xlabel('Final Skill', fontsize=18)
        plt.xlim([0,1])
        # tick-params:
        plt.tick_params(left = True, right = False , labelleft = True)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        if save_location != False:
            plt.savefig(save_location, dpi=512)
        plt.show()
        
    
    def plot_practice_events_histogram(self, colour='blue', n_bins=50, save_location=False):
        plt.figure(figsize=(10, 6))
        plt.hist(self.total_practice_events, bins=[i/n_bins for i in range(n_bins+1)], color=colour, edgecolor='black')
        plt.xlabel('Total Number of Practice Events', fontsize=18)
        # tick-params:
        plt.tick_params(left = True, right = False , labelleft = True)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        if save_location != False:
            plt.savefig(save_location, dpi=512)
        plt.show()

    def plot_trajectory_and_histogram(self, colour_lineplots='Black', colour_histogram='Blue', n_plots=100, n_bins=50, save_location=False):
        # Create figure and axis objects
        fig = plt.figure(figsize=(10, 6))
        grid = plt.GridSpec(1, 2, width_ratios=[2, 1])  # 2:1 ratio for grid width
        
        # Plotting the skill trajectories of first n_plots learners
        ax1 = fig.add_subplot(grid[0])
        for skill_level, prac_times in zip(self.interpolated_skills[:n_plots], self.interpolated_prac_times[:n_plots]):
            ax1.plot(prac_times, skill_level, '-', linewidth=0.5, alpha=0.7, color=colour_lineplots)  # Plot each trajectory
        ax1.set_xlim(0, max([max(time) for time in self.all_practice_times]))  # Set x-axis limit based on maximum time
        ax1.set_ylim(0, 1)
        ax1.set_xlabel('Time', fontsize=22)
        ax1.set_ylabel('Skill',  fontsize=22)

        # Adding deadlines to the plot, if they exist:
        if self.deadlines is not None:
            normalized_weights = [float(i)/max(self.deadline_weights) for i in self.deadline_weights] 
            for deadline, weight in zip(self.deadlines, normalized_weights):
                ax1.axvline(x=deadline, ymin=0, ymax=weight, color='black', alpha=0.5, linestyle='--')
            
        # Creating the histogram on the right
        ax2 = fig.add_subplot(grid[1])
        ax2.hist(self.final_skills, bins=[i/n_bins for i in range(n_bins+1)], density=True, orientation='horizontal', color=colour_histogram, linewidth=0.5)
        ax2.set_ylim(0, 1)
        ax2.yaxis.tick_right() # Move y-axis ticks to the right
        #ax2.yaxis.set_label_position("bottom")
        ax2.set_xlabel('Final Skill', fontsize=19)
        plt.tick_params(left = False, right = True ,  bottom=False, labelbottom=False, labelleft = False)
        #ax2.set_yticks()  # Remove y-axis tick labels
        
        plt.tight_layout()  # Adjust layout to fit
        if save_location != False:
            plt.savefig(save_location, dpi=512)
        plt.show()
        


    def plot_summary_cogsci(self, colour_lineplots='Black', colour_histogram='Blue', n_plots=100, n_bins=50, save_location=False, bw_adjust=1):
        # make dataframe from list of final skills; makes it easier to make the histogram using seaborn
        df_finalS = pd.DataFrame(self.final_skills, columns=['final_skills'])
        
        # Create Figure and Subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'width_ratios': [2, 1]})

        # Plotting the skill trajectories of first n_plots learners
        for skill_level, prac_times in zip(self.interpolated_skills[:n_plots], self.interpolated_prac_times[:n_plots]):
            ax1.plot(prac_times, skill_level, '-', linewidth=0.5, alpha=0.7, color=colour_lineplots)  # Plot each trajectory
        ax1.set_title('Learning Trajectories', fontsize=20)
        ax1.set_xlim(0, max([max(time) for time in self.all_practice_times]))  # Set x-axis limit based on maximum time
        ax1.set_ylim(0, 1)
        ax1.set_xlabel('Time', fontsize=22)
        ax1.set_ylabel('Skill',  fontsize=22)

        # Adding deadlines to the plot, if they exist:
        if self.deadlines is not None:
            normalized_weights = [float(i)/max(self.deadline_weights) for i in self.deadline_weights] 
            for deadline, weight in zip(self.deadlines, normalized_weights):
                ax1.axvline(x=deadline, ymin=0, ymax=weight, color='black', alpha=0.5, linestyle='-', lw=1)
        
        
        # Creating the histogram on the right using seaborn
        sns.kdeplot(df_finalS, ax=ax2, y='final_skills', color=colour_histogram, alpha=0.7, fill=True, bw_adjust=bw_adjust)
        ax2.set_title('Distribution of Final Skills', fontsize=16)
        #ax2.set_ylabel('Final Skill', fontsize=20)  # Label for what was previously the x-axis
        #ax2.set_xlabel('Density', fontsize=22, labelpad=10)  # Label for what was previously the y-axis
        ax2.set_xlabel('')
        ax2.set_ylabel('Skill', fontsize=22)
        ax2.set_xticks([])
        ax2.set_ylim(ax1.get_ylim())  # Match y-limits to line plot y-axis
        
        ax2.yaxis.tick_right()  # Move y-axis ticks to the right
        ax2.yaxis.set_label_position("right")  # Move y-axis label to the right

        
        plt.tight_layout()  # Adjust layout to fit
        if save_location != False:
            plt.savefig(save_location, dpi=512)
        plt.show()        
























