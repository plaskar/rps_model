import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import itertools

from rps_simulation.learning_curves import exponential_learning, logistic_learning
from rps_simulation.forgetting_curves import exponential_forgetting, power_forgetting 
from rps_simulation.practice_rate import simple_linear_rate
from rps_simulation.waiting_times import exponential_waiting_time 
from rps_simulation.rps_base import RPS_Basic, RPS_Basic_Multirun

import time
from IPython.display import display, HTML, clear_output
from rps_simulation import display_progress


class RPS_sensitivity:
    """
    * par_dict dictionary contains all parameters (a,b, alpha, beta, initial_skill) as keys. If corresponding value
        is a number, the parameter is fixed at that number. If the kay-value is a list or 1D array, sensitivity analysisruns 
    """
    def __init__(self, 
                 # contains dict of each parameter and value(s) - a, b, alpha, beta (and optionally c, nu):
                 par_dict={'a': np.round(np.linspace(0, 1, 21), 3), 
                           'b': np.round(np.linspace(0.5, 10, 20), 3),
                           'alpha': 0.4,
                           'beta': 0.3,
                           'initial_skill': 0.1}, 
                 learning='logistic', # other options: 'exponential', 'compound_exp', 'richards' 
                 forgetting='exp', # other options: 'pow' for power forgetting
                 practice_rate='simple', # default is simple_linear_rate, other choice is 'change' 
                 waiting_time= 'exp', # default and only option is exponential waiting time
                 initial_skill=0.1, initial_practice_rate=1):
        
        self.par_dict = par_dict
        self.par_list = list(par_dict.keys()) # all list of all parameters
        self.learning = learning
        self.forgetting = forgetting
        self.practice_rate = practice_rate
        self.waiting_time = waiting_time
        self.initial_practice_rate = initial_practice_rate
        
        # params for which sens. analysis is wanted, along with their number of levels
        self.par_sens = [(key, len(self.par_dict[key])) for key in self.par_list if isinstance(self.par_dict[key], (list, np.ndarray))]
        
        # Making dataframe with parameter values (for adding simulation data to it later)
        temp_vals = [self.par_dict[key] for key, n_levels in self.par_sens]
        temp_comb = list(itertools.product(*temp_vals))
        df = pd.DataFrame(temp_comb, columns=[par[0] for par in self.par_sens])
        
        for key in self.par_list:
            if key not in [par[0] for par in self.par_sens]:
                df[key] = np.repeat(self.par_dict[key], len(df))
        self.df_par = df
        

    def run_sensitivity_analysis(self, quit_thresh=0.04, n_sims=100, max_time=100):
        """
        * The function runs the sensitivity analysis for all combinations of levels of parameters for the 
            rps model as provided in  par_dict
        * After running simulations it adds an attribute 'df_sim' to the RPS_sensitivity class, which stores
            the final_skill, n_prac (number of practice events) and proportion_quit (defined by the quit threshold)
            to df_sim. 
        * 'df_sim' should contain all the data you need from the sensitivity analysis, in case further functionality
            is required for future. 

        TODO: 
        1. add prop_expert to df_sim, calculating the proportion of expertise 9defined using a threshold on skill)
        """
        self.max_time = max_time
        self.n_sims = n_sims
        self.quit_thresh = quit_thresh
        # Initialize df_sim and add appropriate number of rows. 
        # This df will store all relevant data and also be used to plot
        df_sim = pd.DataFrame(np.repeat(self.df_par.values, self.n_sims, axis=0))
        df_sim.columns = self.df_par.columns
        data_sim = {'final_skills': [], 'n_prac': [], 'prop_quit': []} # used to store all sim results 
        
        # run self.n_sims simulations for each combination of parameter values:
        for i in range(len(self.df_par)):
            a_, b_, alpha_, beta_ = self.df_par['a'][i], self.df_par['b'][i], self.df_par['alpha'][i], self.df_par['beta'][i]
            s0_ = self.df_par['initial_skill'][i]
            
            # Define Learning Function
            if self.learning == 'logistic':
                temp_learn = logistic_learning(alpha=alpha_)
            elif self.learning == 'exponential':
                temp_learn = exponential_learning(alpha=alpha_)

            # Defint Forgetting Function
            if self.forgetting=='exp':
                temp_forget = exponential_forgetting(forgetting_rate = beta_)
            elif self.forgetting=='pow':
                temp_forget = power_forgetting(forgetting_rate=beta_)

            # Define practice rate
            if self.practice_rate=='simple':
                temp_pr = simple_linear_rate(a=a_, b=b_)
            
            # Waiting Time Distribution:
            if self.waiting_time=='exp':
                temp_wait = exponential_waiting_time

            # Set up the RPS Basic Multirun:
            temp_sims = RPS_Basic_Multirun(learning_func = temp_learn,
                            forgetting_func = temp_forget,
                            practice_rate_func = temp_pr,
                            waiting_time_dist = temp_wait,
                            n_sims = self.n_sims,
                            initial_skill = s0_, initial_practice_rate=self.initial_practice_rate, max_time=self.max_time)
            # running the sim
            temp_sims.run_multiple_sims()
            
            # adding sim data:
            data_sim['final_skills'] += temp_sims.final_skills
            data_sim['n_prac'] += temp_sims.total_practice_events
            fs = np.array(temp_sims.final_skills) # array of final skills
            prop_quit = np.sum(fs<quit_thresh)/fs.size
            data_sim['prop_quit'].append(prop_quit)

            # Display Progress Bar:
            display_progress(i+1, len(self.df_par))

        df_sim['final_skills'] = data_sim['final_skills']
        df_sim['n_prac'] = data_sim['n_prac']
        df_sim['prop_quit'] = np.repeat(data_sim['prop_quit'], self.n_sims)
        

        # add data to class object
        self.df_sim = df_sim
        self.prop_quit = data_sim['prop_quit']
        self.df_par['prop_quit'] = data_sim['prop_quit']


        
                
    def final_skill_histogram(self, param = 'a',  # parameter for which histogram wanted
                              par_vals = [i/4 for i in range(5)],  # levels of the param
                              par_others ={'b': 5}, # fix values of other params with > 1 level using this dict
                              plot_parms = None # optionally change default plot parameter values
                             ): 
        """
        Plot a histogram of the final skill levels for each param value in par_vals. 
        The other parameters are fixed to the corresponding value in par_others. 
        """

        # Default plot parameters, optionally user can change some/all 
        # of these using by providing plot_parms:
        default_plot_parms = {'alpha': 0.65, 'palette': 'rocket_r', 'bw_adj': 0.5, 'x_fs': 20, 'y_fs': 20, 
                              'title_fs':20, 'legend_head_fs':20, 'legend_txt_fs':20, 'legend_pos':'upper center', 
                              'save_location': None, 'dpi':256
                             }

        # If plot_parms is provided, update default parameters with it
        if plot_parms is not None:
            default_plot_parms.update(plot_parms)

        # Use default_plot_parms for plotting
        plot_parms = default_plot_parms

        # Start with a copy of the simulation dframe. 
        # This will finally become the df we use to make the (kde) histogram:
        df_param = self.df_sim.copy()  
        
        # loop through other parameters and filter on their fixed values:
        for key, value in par_others.items():
            df_param = df_param[df_param[key]==(value)]
            
        # Now filter df_param for the param (Default 'a') parameter being in par_vals list:
        df_param = df_param[df_param[param].isin(par_vals)]

        ###### Make Histogram using seaborn ######
        plt.figure(figsize=(12,8), dpi=128)
        ax = sns.kdeplot(df_param, x='final_skills', hue=param, fill = True, palette=plot_parms['palette'],
                         alpha=plot_parms['alpha'], bw_adjust=plot_parms['bw_adj'])
        plt.title('Effect of $' + param + '$', fontsize=plot_parms['title_fs'])
        plt.xlim([0,1]) # restrict skill range on x-axis
        plt.tick_params(left=False, labelleft=False)
        plt.xlabel('Final Skill', fontsize=plot_parms['x_fs'])
        plt.ylabel('Probability Density', fontsize=plot_parms['y_fs'])

        # make sure to move legend to desired position first, and then change fontsizes:
        sns.move_legend(ax, plot_parms['legend_pos'])
        # set legend font-size:
        plt.setp(ax.get_legend().get_texts(), fontsize=plot_parms['legend_txt_fs']) 
        # set legend heading (title) fontsize:
        plt.setp(ax.get_legend().get_title(), fontsize=plot_parms['legend_head_fs'], text='$'+param+'$', fontweight='bold') 
        

        # # set legend position
        # plt.legend('$'+param +'$', fontsize=plot_parms['legend_txt_fs'], title_fontsize=plot_parms['legend_head_fs'], loc=plot_parms['legend_pos']) 

        
        if plot_parms['save_location']!= None: # save_location given:
            plt.savefig(plot_parms['save_location'], dpi=plot_parms['dpi'])
        plt.show()
        
        
        
        
    
        
    # def practice_events_histogram(self, param = 'a',  # parameter for which histogram wanted
    #                               par_vals = [i/4 for i in range(5)],  # levels of the param
    #                               par_others ={'b': 5}, # fix values of other params with > 1 level using this dict
    #                               plot_parms = None # optionally change default plot parameter values
    #                              ): 
    #     """
    #     Plot a histogram of the total practice-events for each param value in par_vals. 
    #     The other parameters are fixed to the corresponding value in par_others. 
    #     """
        
    #     # Default plot parameters, optionally user can change some/all 
    #     # of these using by providing plot_parms:
    #     default_plot_parms = {'alpha': 0.65, 'palette': 'rocket_r', 'bw_adj': 0.5, 'x_fs': 20, 'y_fs': 20,
    #                           'title_fs':20, 'legend_fs': 18, 'legend_pos': 'upper center', 
    #                           'save_location': None, 'dpi':256 
    #                          }

    #     # If plot_parms is provided, update default parameters with it
    #     if plot_parms is not None:
    #         default_plot_parms.update(plot_parms)

    #     # Use default_plot_parms for plotting
    #     plot_parms = default_plot_parms

    #     # Start with a copy of the simulation dframe. 
    #     # This will finally become the df we use to make the (kde) histogram:
    #     df_param = self.df_sim.copy()  
        
    #     # loop through other parameters and filter on their fixed values:
    #     for key, value in par_others.items():
    #         df_param = df_param[df_param[key]==(value)]
            
    #     # Now filter df_param for the param (Default 'a') parameter being in par_vals list:
    #     df_param = df_param[df_param[param].isin(par_vals)]

    #     # Now make histogram using seaborn:
    #     plt.figure(figsize=(12,8), dpi=128)
    #     ax = sns.kdeplot(df_param, x='n_prac', hue=param, fill = True, palette=plot_parms['palette'],
    #                      alpha=plot_parms['alpha'], bw_adjust=plot_parms['bw_adj'])
    #     plt.title('Effect of $' + param + '$', fontsize=plot_parms['title_fs'])
    #     plt.xlim([0, max(self.df_sim['n_prac'])]) # restrict range on x-axis
    #     plt.tick_params(left=False, labelleft=False)
    #     plt.xlabel('Total Practice Events', fontsize=plot_parms['x_fs'])
    #     plt.ylabel('Probability Density', fontsize=plot_parms['y_fs'])

    #     plt.setp(ax.get_legend().get_texts(), fontsize=plot_parms['legend_fs']) # legend font-size
    #     sns.move_legend(ax, plot_parms['legend_pos'])

    #     if plot_parms['save_location']!= None: # save_location given:
    #         plt.savefig(plot_parms['save_location'], dpi=plot_parms['dpi'])
    #     plt.show()


    def heatmap(self, param_x ='a', param_y='b', save_location=None, dpi=512):
        """
        Makes a heatmap with param_x on x-axis, param_y on y-axis. 
        By default, it shows the prop_quit measure for each combination of parameter values.
        """

        pivot_table = self.df_sim.pivot(index=param_y, columns=param_x, values='prop_quit')

        # Plotting the heatmap
        plt.figure(figsize=(14, 12))
        ax = sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap="coolwarm", annot_kws={"fontsize":7})
        
        ax.figure.axes[-1].set_ylabel('Proportion Quit', size=20) # value bar to side
        plt.gca().invert_yaxis()
        
        # Increase the label size for the color bar
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=12)
        
        plt.title('Heatmap of Quit Proportions', fontsize=19)
        plt.xlabel('Parameter ' + param_x, fontsize=19)
        plt.ylabel('Parameter ' + param_y, fontsize=19)

        if save_location is not None:
            plt.savefig(save_location, dpi=dpi)
        plt.show()

    #def percent_quit_lineplot(self):
        # Plot a line plot of the percentage of learners who quit for each parameter value




















        