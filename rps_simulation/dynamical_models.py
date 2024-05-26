import sympy as sp
#sp.init_printing()
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from . import stability_colour


class logistic_model:
    """
    * This class defines the dynamical RPS model with logistic learning curve.  
    * It can be used to find fixed points (along with stabilities)
    * It can also plot a bifurcation diagram for any parameter over a range of provided values
    """
    
    def __init__(self, a=0.5, b=2, alpha=0.5, beta=0.3):
        S1, a1, b1, alpha1, beta1 = sp.symbols('S,a,b,alpha,beta') # sympy symbols
        
        # adding symbols RHS of diff. eqn. for dS/dt:
        self.S1 = S1; self.a1 = a1; self.b1 = b1; self.alpha1 = alpha1; self.beta1 = beta1
        self.diff_eqn = (-b1 * alpha1) * S1**3 + (b1 - a1) * alpha1 * S1**2 + (a1 * alpha1 - beta1) * S1 

        # parameter values:
        self.a = a; self.b = b; self.alpha = alpha; self.beta=beta

    # Calculating fixed points and stability:
    def find_fixed_points(self):
        # calculate fixed points by setting f(s) = 0
        fp_expr = sp.solve(self.diff_eqn, self.S1) # list of fixed points expressions
        self.fp_expr = fp_expr
        
        # calculating actual fixed point values, using provided param vals (may be complex)
        fp_vals = [  fp.subs([(self.a1, self.a), (self.b1, self.b), (self.alpha1, self.alpha), (self.beta1, self.beta)]) for fp in fp_expr ]
        self.fp_vals = [fp for fp in fp_vals if fp.is_real] # only keep real fixed points
        self.fp_num = len(fp_expr) # number of real fixed-points
          
        # calculating derivatives at only real fixed points:
        self.fp_der_vals = [] # will store derivative of diff_eqn at each fixed point
        self.fp_stab = [] # stores stability (1 = stable, -1 = unstable, 0 = can't say)
        for fp_val in self.fp_vals:
            der_val = sp.diff(self.diff_eqn, self.S1).subs([(self.S1, fp_val), (self.a1,self.a), (self.b1,self.b), (self.alpha1,self.alpha), (self.beta1, self.beta)])
            stability = -1 if der_val>0 else 1 if der_val<0 else 0
            self.fp_der_vals.append(der_val)
            self.fp_stab.append(stability)
            
        # dictionary to be returned
        #out = {'fixed_points': self.fp_vals, 'stability': self.stab_list} 
        out = {'fixed_points': self.fp_vals, 'stability': self.fp_stab, 'derivative': self.fp_der_vals} 
        
        return out


       
        
    def make_bifurcation(self, bf_par='a', par_vals=np.linspace(0, 2, 21), save_location=False ):
        self.find_fixed_points() # start by finding expression for fixed points
        all_pars = {self.a1, self.b1, self.alpha1, self.beta1}
        
        # Define par to be the param for which bif. diag. is asked
        par = self.a1 if bf_par == 'a' else self.b1 if bf_par == 'b' else self.alpha1 if bf_par == 'alpha' else self.beta1
        all_pars.remove(par)  # Remove the bifurcation parameter from the set of all parameters
        
        # Create lambda functions for fixed points and derivatives
        fp_fn = [sp.lambdify(par, fp.subs([(p, getattr(self, p.name)) for p in all_pars]), 'numpy') for fp in self.fp_expr]
        deriv_fp_expr = [sp.diff(self.diff_eqn, self.S1).subs(self.S1, fp) for fp in self.fp_expr]
        deriv_fp_fn = [sp.lambdify(par, deriv_fp.subs([(p, getattr(self, p.name)) for p in all_pars]), 'numpy') for deriv_fp in deriv_fp_expr]
        
        # Calculate numerical values of fixed points and derivatives
        fp_vals = [np.full_like(par_vals, f(par_vals)) for f in fp_fn]
        deriv_fp_vals = [np.full_like(par_vals, f(par_vals)) for f in deriv_fp_fn]
        
        # Determine stability colors
        cols = stability_colour(deriv_fp_vals)
        
        # Plotting
        plt.figure(figsize=(10, 6))
        for i in range(len(fp_vals)):
            plt.scatter(par_vals, fp_vals[i], c=cols[i], s=12)
        
        #plt.title('Bifurcation Diagram', fontsize=19)
        plt.xlabel(f'Parameter {bf_par}', fontsize=19)
        plt.ylim([-0.01, 1])
        plt.xlim(par_vals[0], par_vals[-1])
        plt.ylabel('Equilibrium Skill S', fontsize=19)
        
        # Generate the legend
        legend_elements = [Patch(facecolor='blue', label='Stable'), Patch(facecolor='red', label='Unstable')]
        plt.legend(handles=legend_elements, fontsize=16)
        plt.grid(True)
        if save_location != False:
            plt.savefig(save_location, dpi=512)
        plt.show()




### General Dynamical RPS Learning Function:
class general_model:
    """
    * This class defines the dynamical RPS model with logistic learning curve.  
    * It can be used to find fixed points (along with stabilities)
    * It can also plot a bifurcation diagram for any parameter over a range of provided values
    """
    def __init__(self, diff_eqn, params):
        self.S = sp.Symbol('S')  # Symbol for the state variable S
        self.params = params  # Dictionary of parameters and their default values
        self.diff_eqn = diff_eqn  # Differential equation dS/dt = f(S)

    def find_fixed_points(self):
        # Calculate fixed points by setting f(S) = 0
        fp_expr = sp.solve(self.diff_eqn, self.S)
        self.fp_expr = fp_expr

        # Calculate actual fixed point values using provided parameter values (may be complex)
        fp_vals = [fp.subs([(sp.Symbol(p), v) for p, v in self.params.items()]) for fp in fp_expr]
        self.fp_vals = [fp for fp in fp_vals if fp.is_real]  # Only keep real fixed points
        self.fp_num = len(self.fp_vals)  # Number of real fixed points

        # Calculate derivatives at real fixed points
        self.fp_der_vals = []  # Store derivative of diff_eqn at each fixed point
        self.fp_stab = []  # Store stability (1 = stable, -1 = unstable, 0 = can't say)
        for fp_val in self.fp_vals:
            der_val = sp.diff(self.diff_eqn, self.S).subs([(self.S, fp_val)] + [(sp.Symbol(p), v) for p, v in self.params.items()])
            stability = -1 if der_val > 0 else 1 if der_val < 0 else 0
            self.fp_der_vals.append(der_val)
            self.fp_stab.append(stability)

        # Dictionary to be returned
        out = {'fixed_points': self.fp_vals, 'stability': self.fp_stab, 'derivative': self.fp_der_vals}
        return out

    def make_bifurcation(self, bf_par, par_vals=np.linspace(0, 2, 21), save_location=False, legend_unstable = True):
        self.find_fixed_points() # start by finding expression for fixed points
        
        # Create lambda functions for fixed points and derivatives
        fp_fn = [sp.lambdify(sp.Symbol(bf_par), fp.subs([(sp.Symbol(p), v) for p, v in self.params.items() if p != bf_par]), 'numpy') for fp in self.fp_expr]
        deriv_fp_expr = [sp.diff(self.diff_eqn, self.S).subs(self.S, fp) for fp in self.fp_expr]
        deriv_fp_fn = [sp.lambdify(sp.Symbol(bf_par), deriv_fp.subs([(sp.Symbol(p), v) for p, v in self.params.items() if p != bf_par]), 'numpy') for deriv_fp in deriv_fp_expr]

        # Calculate numerical values of fixed points and derivatives
        fp_vals = [np.full_like(par_vals, f(par_vals)) for f in fp_fn]
        deriv_fp_vals = [np.full_like(par_vals, f(par_vals)) for f in deriv_fp_fn]

        # Determine stability colors
        cols = stability_colour(deriv_fp_vals)

        # Plotting
        plt.figure(figsize=(10, 6))
        for i in range(len(fp_vals)):
            plt.scatter(par_vals, fp_vals[i], c=cols[i], s=12)

        #plt.title('Bifurcation Diagram', fontsize=19)
        plt.xlabel(f'Parameter {bf_par}', fontsize=19)
        plt.ylim([-0.01, 1])
        plt.xlim(par_vals[0], par_vals[-1])
        plt.ylabel('Equilibrium Skill S', fontsize=19)

        # Generate the legend
        if legend_unstable==True:
            legend_elements = [Patch(facecolor='blue', label='Stable'), Patch(facecolor='red', label='Unstable')]
        else:
            legend_elements = [Patch(facecolor='blue', label='Stable')] # don't include unstable in legend
        plt.legend(handles=legend_elements, fontsize=16)
        plt.grid(True)
        if save_location != False:
            plt.savefig(save_location, dpi=512)
        plt.show()


        
        
        


