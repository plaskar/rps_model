
# Different Learning Functions lead to different update rules:
def sigmoid_skill_update(skill):
    alpha = 0.4
    return skill + alpha * skill * (1 - skill)

def exponential_skill_update(skill):
    alpha=0.2
    return skill + alpha*(1-skill)
   
def richards_skill_update(skill, alpha=0.4, nu=1):
    return skill + alpha * (1 - (skill ** nu)) * skill
