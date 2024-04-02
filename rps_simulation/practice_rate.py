"""
* The Practice Rate function takes as input the list of skills (in the basic model). 
* It returns the practice rate, which controls the waiting-time distribution of the next practice-event

# 1. simple_practice_rate:  
    The simple practice rate is a linear function of the skill just after the current practice-event. 
# 2. general_practice_rate: --- to do ----
    
"""



def simple_practice_rate(skill_history, a=0.2, b=5):
    return a + b * skill_history[-1]
