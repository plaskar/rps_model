import numpy as np
#import seaborn as sns
import matplotlib.pyplot as plt



###########################################################################
 ###################### 0. Alias ######################
###########################################################################


    



###########################################################################
 ###################### 1. Concave Learning Curves ######################
###########################################################################

class exponential_learning:
    """
    Standard exponential learning curve.
    Parameters:
        alpha = learning rate
        skill_max = maximum skill (=1 by default)
    Skill is assumed to be positive (>0) with skill_max (=1) the maximum skill level.
    """
    
    def __init__(self, alpha=0.2, skill_max=1):
        self.alpha = alpha
        self.skill_max = skill_max
    
    def updated_skill(self, skill):
        new_skill = skill + self.alpha*(1 - skill/self.skill_max)
        return new_skill

    

class power_learning:
    """
    Power learning curve. Equation is (assuming skill_max =1):
            s(t) = 1 - (1-s0)/(1+t)^mu
    Rate of increas in skill is given by:
            ds/dt = alpha*(1-s)*((1-s)/(1-s0))**(1/alpha)
    
    Parameters:
        s0 = startig skill level.
        alpha = learning rate
        skill_max = maximum skill (=1 by default)
    
    * Skill is assumed to be positive (>0) with skill_max (=1) the maximum skill level.
    * NOTE: Derivative ds/dt depends on starting skill s0, along with current skill s and max-skill s_max
        for the power law of learning. This is unlike exponential and logistic learning curves. 
    """
    
    def __init__(self, alpha=0.2, skill_start=0.1, skill_max=1):
        self.alpha = alpha
        self.skill_max = skill_max
        self.skill_start = skill_start
    
    def updated_skill(self, skill):
        new_skill = skill + self.alpha*(self.skill_max-skill)*((self.skill_max-skill)/(self.skill_max-self.skill_start))**(1/self.alpha)
        return new_skill
    

###########################################################################
 ###################### 2. S-shaped Learning Curves ######################
###########################################################################

class logistic_learning:
    """
    Standard logistic learning curve.
    Parameters:
        alpha = learning rate (=0.4 by default)
        skill_max = maximum skill (=1 by default)
    
    * Skill is assumed to be positive (>0) with skill_max (=1) the maximum skill level.
    * Inflection point is at S = skill_max/2 (= 0.5 by default)
    """
    
    def __init__(self, alpha=0.4, skill_max=1):
        self.alpha = alpha
        self.skill_max = skill_max
    
    def updated_skill(self, skill):
        new_skill = skill + self.alpha*skill*(1 - skill/self.skill_max)
        return new_skill



class richards_learning:
    """
    Richards Learning Curves - a generalization of logistic learning. Rate of increase in skill
    is give by:
                dS/dt = alpha*skill*(1 - (skill/skill_max)**nu )
    Parameters:
        alpha = learning rate (=0.4 by default)
        nu = controls inflection point (nu=1 by Default, making it the same as logistic learning)
        skill_max = maximum skill (=1 by default)
    
    * Skill is assumed to be positive (>0) with skill_max (=1) the maximum skill level.
    * Inflection point is at S = skill_max/2 (= 0.5 by default)
    """
    
    def __init__(self, alpha=0.4, nu = 1, skill_max=1):
        self.alpha = alpha
        self.nu = 1
        self.skill_max = skill_max
     
    def impact_func(self, skill):
        return (self.alpha*self.c) * ((skill/self.skill_max)**(1 - 1/self.c)) * (1 - (skill/self.skill_max)**(1/self.c))

    
    def updated_skill(self, skill):
        new_skill = skill + self.alpha*skill*(1 - (skill/self.skill_max)**self.nu )
        return new_skill
    
    # returns inflection point:
    def inflec_pt(self):
        infl = (1/(1+self.nu))**(1/self.nu)
        return infl

    def plot_impact(self, x_points=np.linspace(0, 1, 201), save_location=False):
        y_points = self.impact_func(x_points)

        # Make and optionally save plot:
        plt.figure(figsize=(8,6))
        plt.plot(x_points, y_points, color='red', lw=2)
        plt.xlabel('Skill', fontsize=16)
        plt.ylabel('Impact of Practice', fontsize=16)
        if save_location != False:
            plt.savefig(save_location, dpi=512)
        plt.show()

class compound_exp_learning:
    """
    Compound Exponential learning curves as described in Murre 2014. Rate of skill increase using
    (default skill_max=1) is:
                dS/dt = (alpha*c) * (S^(1-1/c)) * (1-S^(1/c))
    
    This, like Richard's Curves, it also gives a family of S-shaped curves. Parameter c (>=1) controls the 
    inflection point. c=1 gives back the concave exponential curve. Very large c shifts inflection point 
    upwards towards 1/e = 0.37 (approx.) as c approaches +infinity.
    
    Parameters:
        alpha = learning rate
        c = complexity of task (always >= 1, Default=2). When c=1, we get the concave exponential curve. 
        skill_max = maximum skill (=1 by default)
    """

    def __init__(self, alpha=0.2, c=1, skill_max=1):
        self.alpha = alpha
        self.c = c
        self.skill_max = 1

    def impact_func(self, skill):
        #return (self.alpha*self.c) * ((skill/self.skill_max)**(1 - 1/self.c)) * (1 - (skill/self.skill_max)**(1/self.c))
        return (self.alpha) * ((skill/self.skill_max)**(1 - 1/self.c)) * (1 - (skill/self.skill_max)**(1/self.c))
    
    def updated_skill(self, skill):
        new_skill = skill + self.impact_func(skill)
        return new_skill

    # returns inflection point:
    def inflec_pt(self):
        infl = (1 - 1/self.c)**(self.c)
        return infl


    
    def plot_impact(self, x_points=np.linspace(0, 1, 201), save_location=False):
        y_points = self.impact_func(x_points)

        # Make and optionally save plot:
        plt.figure(figsize=(8,6))
        plt.plot(x_points, y_points, color='red', lw=2)
        plt.xlabel('Skill', fontsize=16)
        plt.ylabel('Impact of Practice', fontsize=16)
        if save_location != False:
            plt.savefig(save_location, dpi=512)
        plt.show()


    # match alpha for provided c2, so that area under impact function
    # remains same as in initialized model
    def match_alpha(self, c2): 
        alpha2 = self.alpha*((2*c2-1)/(2*self.c-1))*(self.c/c2)
        return alpha2
    
    
        
        


    






