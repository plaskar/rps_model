import numpy as np
import matplotlib.pyplot as plt

class RPS_Basic:
    """
    The Basic RPS Model Class has the following features:
    
    1. The skill update function depends generally on the whole learning history
    2. The updated function 
    
    """
    
    def __init__(self, waiting_time_dist, skill_update_func, forgetting_func, practice_rate_func, 
                 initial_skill=0.1, initial_practice_rate=1, max_time=100, forgetting_rate=0.2):

        ## parameters of the RPS_Basic class:
        self.waiting_time_dist = waiting_time_dist
        self.skill_update_func = skill_update_func
        self.forgetting_rate = forgetting_rate
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
                final_skill = self.forgetting_func(current_skill, self.max_time - current_time, self.forgetting_rate)
                final_practice_rate = self.practice_rate_func(self.skill_levels)  # same final_practice rate as at the last practice_event
                self.practice_times.append(self.max_time)
                self.skill_levels.append(final_skill) 
                self.final_skill = final_skill 
                self.practice_rates.append(final_practice_rate); self.final_practice_rate = final_practice_rate
                break
            
            # Calculate skill level just before next practice event
            skill_before_prac = self.forgetting_func(current_skill, wait_time, self.forgetting_rate)
            
            # Calculate skill level just after practice event
            skill_after_prac = self.skill_update_func(skill_before_prac)
            
            # Calculate practice rate for next practice event
            next_practice_rate = self.practice_rate_func(self.skill_levels)
            
            self.practice_times.append(next_prac_time)
            self.skill_levels.append(skill_after_prac)
            self.practice_rates.append(next_practice_rate)

        ## Filling up Summary Attributes
        self.final_skill = self.practice_times[-1]
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
                interpolated_skill = self.skill_levels[i] * np.exp(-self.forgetting_rate * elapsed_time)
                int_practice_times.append(t)
                int_skill_levels.append(interpolated_skill)
        
        # Add the last original points
        int_practice_times.append(self.practice_times[-1])
        int_skill_levels.append(self.skill_levels[-1])

        return int_practice_times, int_skill_levels
    
    
    # Plots a smoothed learning trajectory including the forgetting curves interpolated  between practice-events
    def plot_learning_trajectory(self, least_count=0.01, n_points=10):
        """Plots the cute learning trajectory, including the forgetting phase"""
        interpolated_practice_times, interpolated_skill_levels = self.interpolate_learning_trajectory_dynamic(least_count, n_points)
        
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

    

    

