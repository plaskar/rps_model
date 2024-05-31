import numpy as np 


###########################################################################
 ###################### 1. Basic Forgetting Classes ######################
###########################################################################
class exponential_forgetting:
    """
    Standard exponential forgetting. 
    Inputs:
        * skill (starting skill) & time (elapsed time), 
        * forgetting_rate - rate of Skill decay
        * Smin - the min. value of skill till which forgetting takes place (Default = 0)
    Outputs:
        current skill after forgetting
    """
    def __init__(self, forgetting_rate=0.2, Smin = 0):
        self.forgetting_rate = forgetting_rate
        self.Smin = Smin
    
    # Calculates S(t), given starting skill and time(t):
    def calculate(self, skill, time):
        final_skill = self.Smin + (skill - self.Smin)*np.exp(-self.forgetting_rate*time)
        return final_skill

    # allows updating the forgettin rate, needed to incorporate spacing
    def update_rate(self, new_rate):
        self.forgetting_rate = new_rate

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

    def update_rate(self, new_rate):
        self.forgetting_rate = new_rate

###########################################################################
 ###################### 2. Advanced Forgetting Classes ######################
###########################################################################
class spacing_effect:
    """
    THIS CLASS CALCULATES FORGETTING RATE BASED ON HISTORY OF WAIT TIMES:

    * beta_min -- min. forgetting rate
    * beta_max -- maximum forgetting rate
    * e -- controls how fast forgetting rate decays. Must be >= 0. 
            e=0 means constant forgetting rate (=beta_max)
    * s -- controls non-linearity of spacing effect. (s between 0 and 1)
            s=0 means forgetting rate only decays based on number of
            practice events in the past. 
    """
    def __init__(self, beta_min=0.01, beta_max=0.2, e=1, s=1):
        self.beta_min = beta_min # min. forgetting rate
        self.beta_max = beta_max # max. forgetting rate
        self.e = e # controls how rate of beta (forgetting rate) decay. Higher is faster.
        self.s = s #  s=0 means only number of practice events counts. s=1 linear

    
    def calc_forgetting_rate(self, wait_times):
        """
        wait_times is 1D array or list of length n (>=1) which 
        contains sequence of spacings from 1st to (n+1)-th practice events.

        Returns: beta_current, the calculated forgetting rate till next practice-event
        """
        tmp_exponent = -self.e*np.sum(np.array(wait_times)**self.s)
        beta_current = self.beta_min + (self.beta_max - self.beta_min)*np.exp(tmp_exponent)

        return beta_current
        
        

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





