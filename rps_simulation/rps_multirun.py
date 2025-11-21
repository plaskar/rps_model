import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from rps_simulation.rps_base import RPS_core # importing core model
from rps_simulation.learning_curves import exponential_learning
from rps_simulation.forgetting_curves import exponential_forgetting 
from rps_simulation.practice_rate import simple_linear_rate, tmt_hyperbolic_rate
from rps_simulation.waiting_times import exponential_waiting_time 


##############################################################################
###  Class to have multiple runs of the basic model
##############################################################################

class RPS_multirun:
    """
    Multiple Runs of the RPS_core class and store useful statistics about the simulation.
    Also allows plotting trajectories and final skill histograms, etc. to test how different
    learning and forgetting curves, deadlines, spacings etc. affect results. 
    This class is used to perform sensitivity analysis.
    """
    def __init__(self, waiting_time_dist, learning_func, forgetting_func, practice_rate_func, 
                 deadline_dict = {'deadlines': None, 'deadline_weights': None, 'tmt_effect': None},
                 spacing_func = None,
                 n_sims=1000, initial_skill=0.1, initial_practice_rate=1, max_time=100,
                 interpol_dict=None # optionally define interpolation params
                ):
        
        # Class Attributes:
        self.waiting_time_dist = waiting_time_dist
        self.learning_func = learning_func
        self.forgetting_func = forgetting_func
        self.practice_rate_func = practice_rate_func

        # Interpolation dict:
        default_interpol_dict={'least_count':0.1, 'min_points':5} # default plot interpolation params
        if interpol_dict is not None:
            default_interpol_dict.update(interpol_dict) # update if user provided custon params.
        self.interpol_dict = default_interpol_dict
        
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

    def run_multiple_sims(self, interpolate_forgetting = False):
        """
        This is the main function which runs the self.n_sims simulations and stores the final skill,
        total pratice events, interpolated skill levels during forgetting (warning - this is computationally intensive)    
        """
        
        # set interpolate_forgetting = False to skip this. Makes simulations much faster. 
        # This should be turned False for sensitivity analysis. 
        # Should be True if you want to plot smooth skill trajectories for each individual
        self.interpolate_forgetting = interpolate_forgetting
        
        for i in range(self.n_sims):
            # set up learning curves, forgetting curves and practice rate funcs:
            


            model = RPS_core(waiting_time_dist=self.waiting_time_dist, learning_func=self.learning_func,
                              forgetting_func=self.forgetting_func, practice_rate_func=self.practice_rate_func,
                              deadline_dict = {'deadlines': self.deadlines, 'deadline_weights': self.deadline_weights, 'tmt_effect': self.tmt_effect},
                              spacing_func = self.spacing_func,
                              initial_skill=self.initial_skill, initial_practice_rate=self.initial_practice_rate, max_time=self.max_time)
            model.run_simulation() # run one instance of simulation
            
            # interpolating skills in-between practice events for smooth plots
            # set interpolate_forgetting = False to skip
            if self.interpolate_forgetting: 
                lc, min_pnts = self.interpol_dict['least_count'], self.interpol_dict['min_points']
                interpolated_practice_times, interpolated_skill_levels = model.interpolate_learning_trajectory_dynamic(lc, min_pnts)

                # interpolated data:
                self.interpolated_prac_times.append(interpolated_practice_times)
                self.interpolated_skills.append(interpolated_skill_levels)

            # adding data from current sim
            self.final_skills.append(model.final_skill)
            self.total_practice_events.append(model.total_practice_events)
            self.all_skill_levels.append(model.skill_levels)
            self.all_practice_times.append(model.practice_times)
            self.all_time_lags.append(model.time_lags)
            self.all_forgetting_rates.append(model.forgetting_rates)
            
            
    def plot_final_skill_distribution(self, colour='blue', n_bins=50, bw_adjust=0.5,
                                   save_location=False, save_dpi=512):
        
        plt.figure(figsize=(10, 6))
        
        # Create the distribution plot
        sns.kdeplot(
            data=self.final_skills,
            color=colour,
            fill=True,
            alpha=0.5,
            linewidth=2,
            bw_adjust=bw_adjust
        )
    
        # tick-params:
        plt.tick_params(left = True, right = False , labelleft = True)
        plt.xticks(fontsize=16)
        plt.xlim([0,1])
        plt.yticks([], fontsize=16)
        plt.ylabel('', fontsize=16)
        plt.xlabel('Final Skill', fontsize=16)
        plt.title('Distribution of Final Skills', fontsize=18)

        
        # Add a rug plot at the bottom for data points
        sns.rugplot(
            data=self.final_skills,
            color=colour,
            alpha=0.5,
            height=0.05
        )
        
        if save_location != False:
            plt.savefig(save_location, dpi=save_dpi, bbbox_inches='tight')
        plt.show()
        

    def plot_practice_events_distribution(self, colour='blue', bw_adjust=0.5, save_location=False, save_dpi=512):
        """
        Plot a smooth distribution of total practice events using seaborn's kdeplot.
        
        Parameters:
        -----------
        color : str
            Color of the distribution plot
        bw_adjust : float
            Bandwidth adjustment factor for kernel density estimation
        save_location : str or False
            If provided, saves the plot to this location
        save_dpi : int
            DPI for saved figure
        """
        x_lim = 100*((max(self.total_practice_events)+100)//100) # x-axis limit, rounded to nearest 100
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Create the distribution plot
        sns.kdeplot(
            data=self.total_practice_events,
            color=colour,
            fill=True,
            alpha=0.5,
            linewidth=2,
            bw_adjust=bw_adjust,
            edgecolor='black'
        )
        
        # Customize the plot
        plt.xlabel('Number of Practice Events', fontsize=16)
        plt.ylabel('', fontsize=16)
        plt.title('Distribution of Total Practice Events', fontsize=18 )
        plt.tick_params(left=True, right=False, labelleft=False)
        plt.xlim([0, x_lim])
        plt.yticks([],fontsize=16)
        plt.xticks(fontsize=16)
        
        # Add a rug plot at the bottom for data points
        sns.rugplot(
            data=self.total_practice_events,
            color=colour,
            alpha=0.5,
            height=0.05
        )
        
        # Save if location provided
        if save_location:
            plt.savefig(save_location, dpi=save_dpi, bbox_inches='tight')
        
        plt.show()

    
    def plot_trajectory_and_histogram(self, colour_lineplots='Black', colour_histogram='Blue', n_plots=100, bw_adjust=1, save_location=False, save_dpi=512 ):
        
        # make dataframe from list of final skills; makes it easier to make the histogram using seaborn
        df_finalS = pd.DataFrame(self.final_skills, columns=['final_skills'])
        
        # Create Figure and Subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'width_ratios': [2, 1]})

        # Plotting the skill trajectories of first n_plots learners, if self.interpolate_forgetting = True
        if self.interpolate_forgetting: 
            for skill_level, prac_times in zip(self.interpolated_skills[:n_plots], self.interpolated_prac_times[:n_plots]):
                ax1.plot(prac_times, skill_level, '-', linewidth=0.5, alpha=0.7, color=colour_lineplots)  # Plot each trajectory
        
        # if self.interpolate_trajectories = False, then plot simple individual trajectories  
        else: 
            for skill_level, prac_times in zip(self.all_skill_levels[:n_plots], self.all_practice_times[:n_plots]):
                ax1.plot(prac_times, skill_level, '-', linewidth=0.5, alpha=0.7, color=colour_lineplots)  # Plot each trajectory
            
        ax1.set_title('Learning Trajectories', fontsize=20)
        ax1.set_xlim(0, max([max(time) for time in self.all_practice_times]))  # Set x-axis limit based on maximum time
        ax1.set_ylim(0, 1)
        ax1.set_xlabel('Time', fontsize=22)
        ax1.set_ylabel('Skill',  fontsize=22)

        # Adding deadlines to the plot, if they exist:
        if self.deadlines is not None:
            normalized_weights = [float(i)/sum(self.deadline_weights) for i in self.deadline_weights] 
            for deadline, weight in zip(self.deadlines, normalized_weights):
                ax1.axvline(x=deadline, ymin=0, ymax=weight, color='red', alpha=0.7, linestyle='-', lw=4)
        
        
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
            plt.savefig(save_location, dpi=save_dpi)
        plt.show()        



    def plot_fskill_pe_scatter(self, colour='blue', save_location=False, save_dpi=512):
        """
        Scatter plot of final skill vs total practice events.
        """
        plt.figure(figsize=(8, 6))
        plt.scatter(self.total_practice_events, self.final_skills, s=15, 
                    color=colour, alpha=0.8)
        
        plt.xlabel('Total Practice Events', fontsize=18)
        plt.ylabel('Final Skill', fontsize=18)
        plt.title('Final Skill vs Total Practice Events', fontsize=18)
        x_lim = 100*((max(self.total_practice_events)+100)//100) # x-axis limit, rounded to nearest 100
        plt.xlim(0, x_lim)
        plt.ylim(-0.05, 1)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.grid(True)
        
        if save_location != False:
            plt.savefig(save_location, dpi=save_dpi, bbox_inches='tight')
        plt.show()