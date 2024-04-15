import numpy as np 

###########################################################################
 ###################### 1. Basic Forgetting Classes ######################
###########################################################################

class exponential_forgetting:
    """
    Standard exponential forgetting. 
    Inputs:
    skill (starting skill), time (elapsed time), forgetting_rate
    Outputs:
    current skill after forgetting
    """
    def __init__(self, forgetting_rate=0.2):
        self.forgetting_rate = forgetting_rate
        
    def calculate(self, skill, time):
        final_skill = skill*np.exp(-self.forgetting_rate*time)
        return final_skill

class power_forgetting:
    """
    Standard Power forgetting.
    Inputs:
    skill (starting skill), time (elapsed time), forgetting_rate
    Outputs:
    current skill after forgetting
    """    
    def __init__(self, forgetting_rate=0.2):
        self.forgetting_rate = forgetting_rate
    
    def calculate(self, skill, time):
        final_skill = skill/(1+time)**self.forgetting_rate
        return final_skill

###########################################################################
 ###################### 2. Advanced Forgetting Classes ######################
###########################################################################

############ HELPER FUNCTIONS ############
def forgetting_rate_decreasing(beta_min, beta_max, n_practice, delta=0.1): 
    """
    CHANGING FORGETTING RATES WITH NUMBER OF PRACTICE EVENTS:
    
    Assumption is that forgetting rate lowers (negative-exponentially) from beta_max (starting forgetting rate) 
    to beta_min (min. forgetting rate) as number of practice events n_prac increases. 
    
    How fast forgetting rate decreases is controlled by delta which is non-negative (>=0):
    * delta=0 means forgetting rate is constant (and = beta_max)
    * very larde delta (eg 1 trillion) means already after practicing once, forgetting rate = beta_min
    * Default delta = 0.1 means after 10 practice events, forgetting rate moves 36% towards beta_min from beta_max 

    The function returns the forgetting rate, given inputs beta_min, beta_max, n_practice and (optionally) delta
    """
    return beta_min + (beta_max - beta_min)*np.exp(-delta*n_practice)



