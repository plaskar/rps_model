import numpy as np

"""
* The Practice Rate function takes as input the list of skills (in the basic model). 
* It returns the practice rate, which controls the waiting-time distribution of the next practice-event

# 1. simple_practice_rate:  
    The simple practice rate is a linear function of the skill just after the current practice-event. 

# 2. general_practice_rate: --- to do ----    
"""


def simple_linear_rate(skill_history, a=0.2, b=5):
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
    practice_rate = a + b*skill_history[-1]
    return practice_rate


def linear_rate_plus_change(skill_history, a=0.2, b=5, c=1, min_rate=0.01):
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
    if len(skill_history) >= 3: # at least 2 practice events
        practice_rate = max(a + b*skill_history[-1] + c*(skill_history[-1] - skill_history[-2]), min_rate)
    else: # if only 1 practice event so far
        practice_rate = a + b*skill_history[-1]
    
    return practice_rate


def general_linear_rate(skill_history, weights, min_rate=0.01):
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
    
    # converting to numpy array (in case input is python list)
    w = np.array(weights)
    s = np.array(skill_history)
    
    length = min(len(skill_history), len(weights)-1)  # number of terms in skill_array to consider 
    
    practice_rate = max(np.sum(np.flip(s[-length:])*w[1:]) + w[0], min_rate)
    
    return practice_rate







    




