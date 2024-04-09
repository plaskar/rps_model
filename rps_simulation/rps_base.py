import numpy as np
import matplotlib.pyplot as plt

from rps_simulation.learning_curves import sigmoid_skill_update, exponential_skill_update, richards_skill_update 
from rps_simulation.forgetting_curves import exponential_forgetting 
from rps_simulation.practice_rate import simple_linear_rate
from rps_simulation.waiting_times import exponential_waiting_time 


##############################################################################
### 1. RPS_Basic Class runs one instant of the model, giving the learning trajectory
##############################################################################
class RPS_Basic:
    """
    The Basic RPS Model Class makes one run of the model. 
    
    * It has a fixed forgetting rate, fixed learning rate. 
    
    * Users have to specify:
        1. "waiting_time_dist": the waiting time distribution
        2. "skill_update_function": this should tell us how to update the skill, based on current skill
        3. "forgetting_func": the forgetting function - exponential, power or something else. 
            It will use the forgetting_rate which is fixed
        4. "practice_rate_func": This should provide a positive practice rate (which controls wiating time)
    
    * The practice function takes as input the skill_levels history so far and generates an output practice rate
    For the basic model, this is simply rate = a + b*skill_level[-1]
    
    * Multiple runs of the simulation for a fixed learning curve will be done in the next class
    """
    
    def __init__(self, 
                 skill_update_func=exponential_skill_update, # by default, we have exponential update s_new = s_old + alpha*(1-s_old) 
                 forgetting_func=exponential_forgetting, # default is exponential forgetting
                 practice_rate_func=simple_linear_rate, # default is simple_linear_rate 
                 waiting_time_dist = exponential_waiting_time, # default is exponential (NOT Pareto) waiting times 
                 initial_skill=0.1, initial_practice_rate=1, max_time=100):

        ## parameters of the RPS_Basic class:
        self.waiting_time_dist = waiting_time_dist
        self.skill_update_func = skill_update_func
        #self.forgetting_rate = forgetting_rate
        self.forgetting_func = forgetting_func
        self.practice_rate_func = practice_rate_func

        # starting values and time_window
        self.initial_skill = initial_skill
        self.initial_practice_rate = initial_practice_rate
        self.max_time = max_time
        
        # Initialize empty lists for simulation results
        self.practice_times = []
        self.skill_levels = []
        self.practice_rates = []
            
        # Summary attributes: 
        self.final_skill = None # final skill at t = max_time, the end of the time-window
        self.final_practice_rate = None # final_practice_rate = practice_rate
        self.total_practice_events = None # total practice events during the run
        self.time_lags = [] # list of time_lags between practice events. Length = (# of practice_events) - 1
        

    # Generates the run_data = which includes practice_times, skill_levels and practice_rates attributes of the class
    def run_simulation(self):
        self.practice_times = [0]
        self.skill_levels = [self.initial_skill]
        self.practice_rates = [self.initial_practice_rate]
        
        while self.practice_times[-1] < self.max_time:
            current_time = self.practice_times[-1]
            current_skill = self.skill_levels[-1]
            current_practice_rate = self.practice_rates[-1]
            
            # Calculate time until next practice event
            wait_time = self.waiting_time_dist(current_practice_rate)
            next_prac_time = current_time + wait_time

            # If next practice time is beyond max_time, calculate final skill level:
            if next_prac_time > self.max_time:
                final_skill = self.forgetting_func(current_skill, self.max_time - current_time)
                final_practice_rate = self.practice_rate_func(self.skill_levels)  # same final_practice rate as at the last practice_event
                self.practice_times.append(self.max_time)
                self.skill_levels.append(final_skill) 
                self.final_skill = final_skill 
                self.practice_rates.append(final_practice_rate); self.final_practice_rate = final_practice_rate
                break
            
            # Calculate skill level just before next practice event
            skill_before_prac = self.forgetting_func(current_skill, wait_time)
            
            # Calculate skill level just after practice event
            skill_after_prac = self.skill_update_func(skill_before_prac)
            
            # Calculate practice rate for next practice event
            next_practice_rate = self.practice_rate_func(self.skill_levels)
            
            self.practice_times.append(next_prac_time)
            self.skill_levels.append(skill_after_prac)
            self.practice_rates.append(next_practice_rate)

        ## Filling up Summary Attributes
        self.final_skill = self.skill_levels[-1]
        self.final_practice_rate = self.practice_rates[-1]
        self.total_practice_events = len(self.practice_times) - 2
        self.time_lags = [ self.practice_times[i+1] - self.practice_times[i] for i in range(1,self.total_practice_events) ]

        return self

    # Returns data from the simulation-run
    def data(self): 
        # Constructing summary attributes dictionary to return
        summary_attributes = {'final_skill': self.final_skill,
                              'final_practice_rate': self.final_practice_rate,
                              'total_practice_events': self.total_practice_events,
                              'time_lags': self.time_lags,
                              'practice_times': self.practice_times,
                              'skill_levels': self.skill_levels,
                              'practice_rates': self.practice_rates
                             }
        return summary_attributes

    def plot_simple_trajectory(self):
        """plot simple learning trajectory without the forgetting-bits:"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.practice_times, self.skill_levels, marker='o', linestyle='-', color='#FF6B6B')
        #plt.plot(interpolated_practice_times, interpolated_skill_levels, linestyle='-', color='Black')
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
                # Otherwise, divide the gap by the least_count to determine the number of points. Min. min_points are added
                points_count = min(int(time_gap / least_count), min_points)  # Ensure at least min_points interpolated points
                times = np.linspace(start_time, end_time, points_count, endpoint=False)
            
            for t in times[:-1]:  # Exclude the last point to prevent overlap
                elapsed_time = t - start_time
                #interpolated_skill = self.skill_levels[i] * np.exp(-self.forgetting_rate * elapsed_time)
                interpolated_skill = self.forgetting_func(self.skill_levels[i], elapsed_time)
                int_practice_times.append(t)
                int_skill_levels.append(interpolated_skill)
        
        # Add the last original points
        int_practice_times.append(self.practice_times[-1])
        int_skill_levels.append(self.skill_levels[-1])

        return int_practice_times, int_skill_levels
    
    
    # Plots a smoothed learning trajectory including the forgetting curves interpolated  between practice-events
    def plot_learning_trajectory(self, least_count=0.01, min_points=10):
        """Plots the cute learning trajectory, including the forgetting phase"""
        interpolated_practice_times, interpolated_skill_levels = self.interpolate_learning_trajectory_dynamic(least_count, min_points)
        
        plt.figure(figsize=(10, 6))
        # plt.plot(interpolated_practice_times, interpolated_skill_levels, marker='o', linestyle='-')
        plt.plot(interpolated_practice_times, interpolated_skill_levels, linestyle='-', color='#FF6B6B')
        plt.title('Learning Trajectory with Interpolation', fontsize=22)
        plt.xlabel('Practice Time', fontsize=18)
        plt.ylabel('Skill Level', fontsize=18)
        plt.grid(True, alpha=1)
        plt.show()
    
    # Plots practice_times_plot - each vertical line corresponds to a practice-event
    def practice_times_plot(self):
        """Plot practice times as vertical lines."""
        plt.figure(figsize=(10, 2))  # Wide and not too high
        for prac_time in self.practice_times[1:-1]:  # Excluding the first and last elements
            plt.axvline(x=prac_time, color='black', linestyle='-', linewidth=2)
        plt.title('Practice Times', fontsize=22)
        plt.xlabel('Time', fontsize=18)
        plt.yticks([])  # Hide y-axis ticks
        plt.tight_layout()
        plt.show()



##############################################################################
### 2. Class to have multiple runs of the basic model
##############################################################################
class RPS_Basic_Multirun:
    """
    Multiple Runs of the RPS_Basic class.
    """
    def __init__(self, waiting_time_dist, skill_update_func, forgetting_func, practice_rate_func, 
                 n_sims=1000, initial_skill=0.1, initial_practice_rate=1, max_time=100):
        # Class Attributes:
        self.waiting_time_dist = waiting_time_dist
        self.skill_update_func = skill_update_func
        self.forgetting_func = forgetting_func
        self.practice_rate_func = practice_rate_func
        
        self.n_sims = n_sims  # Number of simulations to run
        self.initial_skill = initial_skill
        self.initial_practice_rate = initial_practice_rate
        self.max_time = max_time
        #self.forgetting_rate = forgetting_rate
        
        # Summary Data from all Runs:
        self.final_skills = []  # To store final skill levels from all sims
        self.practice_events_counts = []  # To store the number of practice events from all sims
        self.all_skill_levels = [] # list of lists, contains skill_levels for each simulation run
        self.all_practice_times = [] # contains practice_time list for each simulation run
        self.all_practice_rates = [] # contains practice_time list for each simulation run
        
        self.interpolated_skills = []
        self.interpolated_prac_times = []

    def run_multiple_sims(self):
        for _ in range(self.n_sims):
            model = RPS_Basic(waiting_time_dist=self.waiting_time_dist, skill_update_func=self.skill_update_func,
                              forgetting_func=self.forgetting_func, practice_rate_func=self.practice_rate_func,
                              initial_skill=self.initial_skill, initial_practice_rate=self.initial_practice_rate, max_time=self.max_time)
            model.run_simulation()
            
            interpolated_practice_times, interpolated_skill_levels = model.interpolate_learning_trajectory_dynamic(least_count=0.01, min_points=10)
            
            self.final_skills.append(model.final_skill)
            self.practice_events_counts.append(model.total_practice_events)
            self.all_skill_levels.append(model.skill_levels)
            self.all_practice_times.append(model.practice_times)
            
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
        plt.hist(self.practice_events_counts, bins=[i/n_bins for i in range(n_bins+1)], color=colour, edgecolor='black')
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
        grid = plt.GridSpec(1, 2, width_ratios=[2, 1])  # 4:1 ratio for grid width
        
        # Plotting the skill trajectories of first n_plots learners
        ax1 = fig.add_subplot(grid[0])
        for skill_level, prac_times in zip(self.interpolated_skills[:n_plots], self.interpolated_prac_times[:n_plots]):
            ax1.plot(prac_times, skill_level, '-', linewidth=0.5, alpha=0.7, color=colour_lineplots)  # Plot each trajectory
        ax1.set_xlim(0, max([max(time) for time in self.all_practice_times]))  # Set x-axis limit based on maximum time
        ax1.set_ylim(0, 1)
        ax1.set_xlabel('Time', fontsize=19)
        ax1.set_ylabel('Skill',  fontsize=19)
        
        # Creating the histogram on the right
        ax2 = fig.add_subplot(grid[1])
        ax2.hist(self.final_skills, bins=[i/n_bins for i in range(n_bins+1)], orientation='horizontal', color=colour_histogram, linewidth=0.5)
        ax2.set_ylim(0, 1)
        ax2.yaxis.tick_right() # Move y-axis ticks to the right
        ax2.set_xlabel('Final Skill', fontsize=19)
        plt.tick_params(left = False, right = False , labelleft = False)
        #ax2.set_yticks()  # Remove y-axis tick labels
        
        plt.tight_layout()  # Adjust layout to fit
        if save_location != False:
            plt.savefig(save_location, dpi=512)
        plt.show()
        

    

