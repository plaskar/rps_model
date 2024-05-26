import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import seaborn as sns
import itertools

from rps_simulation.learning_curves import exponential_learning, logistic_learning
from rps_simulation.forgetting_curves import exponential_forgetting, power_forgetting 
from rps_simulation.practice_rate import simple_linear_rate
from rps_simulation.waiting_times import exponential_waiting_time 
from rps_simulation.rps_base import RPS_Basic, RPS_Basic_Multirun


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
                temp_wait = exponential_waiting_time`

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
            data_sim['n_prac'] += temp_sims.practice_events_counts
            fs = np.array(temp_sims.final_skills) # array of final skills
            prop_quit = np.sum(fs<quit_thresh)/fs.size
            data_sim['prop_quit'].append(prop_quit)
            

        df_sim['final_skills'] = data_sim['final_skills']
        df_sim['n_prac'] = data_sim['n_prac']
        df_sim['prop_quit'] = np.repeat(data_sim['prop_quit'], self.n_sims)
        

        # add data to class object
        self.df_sim = df_sim
        self.prop_quit = data_sim['prop_quit']
        self.df_par['prop_quit'] = data_sim['prop_quit']


        
                
    def final_skill_histogram(self, param = 'a',  # parameter for which histogram wanted
                              par_vals = [i/4 for i in range(5)],  # levels of the param
                              par_others ={'b': 5}): # fix values of other params with > 1 level using this dict
        """
        Plot a histogram of the final skill levels for each value in par_vals
        """
        # getting list of other prameters which had more than 1 level in the sens. analysis:
        par_sens_other = [key for key,val in self.par_sens if key!=param] 
        #par_sens_others = [key for key in ]
        #mask ==
        df_tmp = self.df_sim(self.df_sim[param]
                            ) # 
        
        
        
        
        
    #def practice_events_histogram(self):
        # Plot a histogram of the total practice events for each parameter value
        
    #def percent_quit_lineplot(self):
        # Plot a line plot of the percentage of learners who quit for each parameter value




















        