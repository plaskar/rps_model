import sympy as sp
import numpy as np
import matplotlib.pyplot as plt


## ---- Defining iterated maps -----
def f_con(x, alpha=0.2, beta=0.2, a=0.2, b=5):
    x1 = alpha + (1-alpha)*x*np.exp(-beta/(a + b*x))
    return x1

def f_sig(x, alpha=0.2, beta=0.2, a=0.2, b=5):
    x1 = (1+alpha)*x*np.exp(-beta/(a+b*x)) - alpha*(x**2)*np.exp(-2*beta/(a+b*x))
    return x1 

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



def create_bifurcation_diagram(param_range, param_name='a', n_points=100):
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
    fixed_points = []
    stability = []
    
    for param in param_values:
        # Create function with current parameter value
        if param_name == 'a':
            f = lambda x: f_sig(x, a=param)
        else:
            f = lambda x: f_sig(x, b=param)
            
        # Find fixed points for this parameter value
        fps = find_all_fixed_points(f)
        
        # Store fixed points and their stability
        for fp in fps:
            deriv, _ = check_stability(f, fp)
            fixed_points.append([param, fp])
            stability.append('stable' if abs(deriv) < 1 else 'unstable')
    
    fixed_points = np.array(fixed_points)

    # Plot bifurcation diagram
    plt.figure(figsize=(12, 8))
    
    # Plot stable points in blue, unstable in red
    stable_mask = np.array(stability) == 'stable'
    
    plt.plot(fixed_points[stable_mask, 0], fixed_points[stable_mask, 1], 
             'b.', label='Stable', markersize=1)
    plt.plot(fixed_points[~stable_mask, 0], fixed_points[~stable_mask, 1], 
             'r.', label='Unstable', markersize=1)
    
    plt.xlabel(f'Parameter {param_name}')
    plt.ylabel('Fixed Point Value')
    plt.title(f'Bifurcation Diagram for Parameter {param_name}')
    plt.legend()
    #plt.grid(True)
    plt.show()
