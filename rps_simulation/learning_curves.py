
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
    
    def updated_skill(self, skill):
        new_skill = skill + self.alpha*skill*(1 - (skill/self.skill_max)**nu )
        return new_skill

    






