import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve


## ---- Defining iterated maps -----
def f_con(x, alpha=0.2, beta=0.3, a=0.2, b=2):
    x1 = alpha + (1-alpha)*x*np.exp(-beta/(a + b*x))
    return x1

def f_sig(x, alpha=0.4, beta=0.3, a=0.2, b=2):
    x1 = (1+alpha)*x*np.exp(-beta/(a+b*x)) - alpha*(x**2)*np.exp(-2*beta/(a+b*x))
    return x1 

# ----- Finding FPs and Bif Diags. -----
def find_all_fixed_points(f, x_range=(0,1), n_guesses=100, tolerance=1e-6):
    """
    Find all fixed points of function f in given range using multiple initial guesses.
    
    Parameters:
    -----------
    f : function
        The iterated map function
    x_range : tuple
        Range to search for fixed points (min, max)
    n_guesses : int
        Number of initial guesses to try
    tolerance : float
        Tolerance for considering a point as fixed point
        
    Returns:
    --------
    fixed_points : list
        List of all unique fixed points found
    """
    # Create array of initial guesses
    x_guesses = np.linspace(x_range[0], x_range[1], n_guesses)
    fixed_points = []
    
    # Function to find zeros of f(x) - x
    def fixed_point_equation(x):
        return f(x) - x
    
    # Try each initial guess
    for x0 in x_guesses:
        sol = fsolve(fixed_point_equation, x0)
        # Check if it's actually a fixed point and within range
        if (abs(fixed_point_equation(sol)) < tolerance and 
            x_range[0] <= sol <= x_range[1]):
            fixed_points.append(sol[0])
    
    # Remove duplicates within tolerance
    unique_fps = []
    for fp in sorted(fixed_points):
        if not unique_fps or min(abs(np.array(unique_fps) - fp)) > tolerance:
            unique_fps.append(fp)
            
    return np.array(unique_fps)


# Check stability of each fixed point
def check_stability(f, x, h=1e-7):
    """Calculate derivative at fixed point and determine stability"""
    deriv = (f(x + h) - f(x - h))/(2*h)
    return deriv, 'stable' if abs(deriv) < 1 else 'unstable'



#### ------ MAIN FUNCTION ---------------
def create_bifurcation_diagram(param_range, param_name='a', 
                               lc='sig', n_points=100, save=False):
    """
    Create bifurcation diagram by tracking fixed points as parameter varies.
    
    Parameters:
    -----------
    param_range : tuple
        Range of parameter values (min, max)
    param_name : str
        Parameter to vary ('a' or 'b')
    n_points : int
        Number of parameter values to try
    """
    param_values = np.linspace(param_range[0], param_range[1], n_points)

    stab1_x = []; stab1_y = []
    stab2_x = []; stab2_y = []
    unstab_x = []; unstab_y = []
    
    for param in param_values:
       
        # If Learning curve is sigmoid:
        if lc == 'sig':
            if param_name == 'a':
                f = lambda x: f_sig(x, a=param, b=2)
                fps = find_all_fixed_points(f)
            else:
                f = lambda x: f_sig(x, b=param, a=0.2)
                fps = find_all_fixed_points(f)
            
            # Now store the fixed point branches
            if len(fps) == 1: # only 0 is stable fp
                stab1_x.append(param); stab1_y.append(fps[0])
                
            elif len(fps) == 3: # 3 fps - 2 stab, 1 unstab
                stab1_x.append(param); stab1_y.append(fps[0])
                stab2_x.append(param); stab2_y.append(fps[2])
                unstab_x.append(param); unstab_y.append(fps[1])
            
            else: # 2 fps - 0-unstable + high stable
                unstab_x.append(param); unstab_y.append(fps[0])
                stab2_x.append(param); stab2_y.append(fps[1])
            
            

        # If learning curve is concave:
        if lc=='con':
            if param_name == 'a':
                f = lambda x: f_con(x, a=param, b=2)
                fps = find_all_fixed_points(f)
            else:
                f = lambda x: f_con(x, b=param, a=0.2)
                fps = find_all_fixed_points(f)
            
            stab1_x.append(param); stab1_y.append(fps[0]) # only 1 stable branch for exp case

        

    # Plot bifurcation diagram
    plt.figure(figsize=(12, 8))

    if lc =='sig': # plotting sigmoid bifs:
        plt.plot(stab1_x, stab1_y, lw=4, color='black', linestyle='-', label='Stable')
        plt.plot(stab2_x, stab2_y, lw=4, color='black', linestyle='-')
        plt.plot(unstab_x, unstab_y, lw=4, color='black', linestyle=':', label='Unstable')
    
    elif lc =='con': # plotting concave bifs:
        plt.plot(stab1_x, stab1_y, lw=4, color='black', linestyle='-', label='Stable')
        
    if param_name=='a':
        plt.xlabel('Parameter $\mathit{a}$', fontsize=19)
    elif param_name=='b':
        plt.xlabel('Parameter $\mathit{b}$', fontsize=19)
    
    plt.ylabel('Skill (S) Fixed Points', fontsize=19)
    plt.legend(fontsize=16)
    
    if save is not False:
        plt.savefig(save, dpi=512)
        
    #plt.grid(True)
    plt.show()
