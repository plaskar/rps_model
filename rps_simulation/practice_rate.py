import numpy as np

"""
* The Practice Rate function takes as input the list of skills (in the basic model). 
* It returns the practice rate, which controls the waiting-time distribution of the next practice-event

# 1. simple_practice_rate:  
    The simple practice rate is a linear function of the skill just after the current practice-event. 

# 2. general_practice_rate: --- to do ----    
"""

class simple_linear_rate:
    """
    Simple Practice Rate Equation in Basic RPS Model:
            practice_rate = a + b*S
    Here: 
        S = the current skill (S) just after practice (the increased skill)
        a = min. practice rate
        b = sensitivity to success
    
    Inputs: 
        skill_history - the list of skill_values containing starting_skill (S0) and skill S just after every practice event
        Optionally, a and b are inputs as well. 
    Output:
        The practice rate (must be positive)
    """
    
    def __init__(self, a=0.2, b=5):
        self.a= a
        self.b = b

    def calculate(self, skill_history):
        practice_rate = self.a + self.b*skill_history[-1]
        return practice_rate


class linear_rate_plus_change:
    """
    practice rate equation now has an additional term c*(S - S_prev):
                practice_rate = a + b*S + c*(S - S_prev)

    Here: 
    S = the current skill (S) just after practice (the increased skill)
    a = min. practice rate
    b = sensitivity to success

    Inputs: 
        skill_history: the list of skill_values containing starting_skill (S0) and skill S just after every practice event

    The idea is that learners have a higher practice rate (are more motivated) when their current skill 
    is more than their skill the last time they practiced. If their current skill is less than before 
    c*(S - S_prev) is negative and the practice rate i lower than before. 
    
    NOTE: Since practice rate must always be positive, there is a hard lower limit of min_rate built in which must be positive
    """
    
    def __init__(self, a=0.2, b=5, c=1, min_rate=0.01):
        self.a = a
        self.b = b
        self.c = c
        self.min_rate = min_rate
        
    def calculate(self, skill_history):
        if len(skill_history) >= 3: # at least 2 practice events
            practice_rate = max(self.a + self.b*skill_history[-1] + self.c*(skill_history[-1] - skill_history[-2]), self.min_rate)
        else: # if only 1 practice event so far
            practice_rate = self.a + self.b*skill_history[-1]
        return practice_rate        


class general_linear_rate:
    """
    In both the cases above, the practice_rate is a linear function of skill history (call it 's'):
    
        simple_linear_rate: 
                practice_rate = a + b*s[-1]
        
        linear_rate_plus_change: 
            here the equation is the same as the following (after some algebra):
                practice_rate = a + (b+c)*s[-1] + (-c)*s[-2]
        
    This function takes as inputs: 
        skill_history: the list of skill_values containing starting_skill (S0) and skill S just
                       after every practice event
        weights = [w0, w1, ... wn], a list of length n+1 (say), with n >= 1
                   w0 is the intercept (constant term), with w0=a in the two cases above.
                   w1 is the weight given to skill_history[-1], the skill after the most recent practice effect
                   w2 weight given to skill_history[-2], the skill after last practice event and so on....
    
    """    
    def __init__(self, weights, min_rate=0.01):
        self.weights =weights
        self.min_rate = min_rate

    def calculate(self, skill_history):
        # converting to numpy array (in case input is python list)
        w = np.array(self.weights)
        s = np.array(skill_history)
        
        length = min(len(s), len(w)-1)  # number of terms in skill_array to consider
        practice_rate = max(np.sum(np.flip(s[-length:])*w[1:]) + w[0], self.min_rate)
        return practice_rate


###########################################################################
 #################### 2. ADDING EFFECT OF DEADLINES (TMT) #################
###########################################################################

class tmt_hyperbolic_rate:
    def __init__(self, impulsivity=1):
        self.impulsivity = impulsivity

    def calculate(self, deadlines, deadline_weights, curr_time, skill_hist):
        deadlines_left = [t for t in deadlines if t > curr_time]

        # if no deadlines left, no additive effect of deadlines
        if len(deadlines_left) == 0: 
            return 0
        else: # at least 1 deadline left
            next_deadline = min(deadlines_left) # time of next deadline
            next_wgt = deadline_weights[deadlines.index(next_deadline)] # weight of next deadline

            # calculate and return increased prac. rate due to deadline (tmt_effect)
            tmt_effect = (skill_hist[-1]*next_wgt)/(1 + self.impulsivity*(next_deadline - curr_time))
            
            return tmt_effect
            
        
        
        
        
        

    




