import sympy as sp
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


       
        
    def make_bifurcation(self, bf_par='a', par_vals=np.linspace(0, 2, 21), save_location=False, bw_paper = False ):
        """
        Makes colored bifurcation diagrams of stable/unstable fixed points. 
        
        Params:
        * bf_par =  the parameter for which bifurcation diagram is wanted
        * par_vals = 1D np.array or List with values of the parameter for which fixed points are plotted
        * save_location =  if not False, the generated plot is saved locally at the given local location
        * bw_paper = Make black-and-white plot for the paper

        """
        
        self.find_fixed_points() # start by finding expression for fixed points
        all_pars = {self.a1, self.b1, self.alpha1, self.beta1}
        
        # Define par to be the param for which bif. diag. is needed
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

        # ----------------
        # --- Plotting ---
        # ----------------
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

        # uncomment below if you want to get calculated fixed point and derivatives:
        return fp_vals, deriv_fp_vals  



    """
    Make bifurcation diagram for parameter a
    """
    def make_bif_a(self, a_max = 3, n_vals=2001, save_location=False):
        a1 = (4*self.b*self.beta/self.alpha)**(1/2) - self.b  # point where 2 more fps emerge
        a2 = self.beta/self.alpha  # pont where S=0 is becomes unstable

        stab1_x = np.linspace(0, a2, n_vals)  # stable S=0 branch 
        stab2_x = np.linspace(a1, a_max, n_vals)  # stable top branch
        unstab_x = np.linspace(a1, a_max, n_vals) # unstable branch
        

        stab1_y = [0 for i in range(n_vals)] # get y-vals for stable dropout branch
        stab2_y = [ (self.b- a_)/(2*self.b) + np.sqrt(self.alpha*(self.alpha*(a_ + self.b)**2 - 4*self.b*self.beta))/(2*self.alpha*self.b) for a_ in stab2_x ] # get y-vals for expertise branch
        # unstab_y = [ (self.b- a_)/2*self.b - np.sqrt(self.alpha*(self.alpha*(a_ + self.b)**2 - 4*self.b*self.beta))/(2*self.alpha*self.b) for a_ in unstab_x if (a_ < a2) else 0] # get y-vals for unstable branch
        unstab_y = [
            (self.b - a_)/(2 * self.b) - np.sqrt(self.alpha * ((self.alpha * (a_ + self.b) ** 2) - 4 * self.b * self.beta)) / (2 * self.alpha * self.b)
            if (a_ < a2) else 0 for a_ in unstab_x
            ]


        # ----------------
        # --- Plotting ---
        # ----------------
        plt.figure(figsize=(10, 6))

        # Plot the 3 branches:
        plt.plot(stab1_x, stab1_y, lw=4, color='black', linestyle='-', label='Stable')
        plt.plot(stab2_x, stab2_y, lw=4, color='black', linestyle='-', label='_nolegend_')
        plt.plot(unstab_x, unstab_y, lw=4, color='black', linestyle=':', label='Unstable')
        
        # Add the shaded region between a1 and a2
        plt.axvspan(a1, a2, color='black', alpha=0.15, label='Hysteresis Zone')  # Translucent black region
        
        # Add vertical lines at a1 and a2
        plt.axvline(x=a1, lw=1, color='black', linestyle='--')
        plt.axvline(x=a2, lw=1, color='black', linestyle='--')

        # Add arrows showing flow directions
        arrow_gap = 0.15
        arrow_positions_x = [np.linspace(a1, a2, 3)[1]] # Positions for arrows along x-axis
        arrow_positions_y_up =  [- arrow_gap + (self.b - a_)/(2 * self.b) - np.sqrt(self.alpha * ((self.alpha * (a_ + self.b) ** 2) - 4 * self.b * self.beta))/ (2 * self.alpha * self.b) for a_ in arrow_positions_x]
        arrow_positions_y_down =  [ arrow_gap + (self.b - a_)/(2 * self.b) - np.sqrt(self.alpha * ((self.alpha * (a_ + self.b) ** 2) - 4 * self.b * self.beta))/ (2 * self.alpha * self.b) for a_ in arrow_positions_x]
        

        for i, x in enumerate(arrow_positions_x):
            # Arrow pointing up (above the unstable branch)
            plt.annotate(
                '', 
                xy=(x, arrow_positions_y_up[i]), 
                xytext=(x, arrow_positions_y_up[i] + arrow_gap),
                arrowprops=dict(facecolor='black', shrink=0.2, width=1, headwidth=8)
            )
        
            # Arrow pointing down (below the unstable branch)
            plt.annotate(
                '', 
                xy=(x, arrow_positions_y_down[i]), 
                xytext=(x, arrow_positions_y_down[i] - arrow_gap),
                arrowprops=dict(facecolor='black', shrink=0.2, width=1, headwidth=8)
            )

        
        #plt.title('Bifurcation Diagram', fontsize=19)
        plt.xlabel('Parameter $\mathit{a}$', fontsize=19)
        plt.ylim([-0.02, 1])
        plt.xlim(0, a_max)
        plt.ylabel('Skill (S) Fixed Points', fontsize=19)

        # Add x-ticks including a1 and a2
        xticks = [0, 1, 2, 3, a1, a2]  # Define ticks
        plt.xticks(xticks)  # Set the tick positions
        
        # Customize all ticks, set labels for numerical ticks
        ax = plt.gca()
        labels = [
            '' if tick == a1 or tick == a2 else f'{tick:.1f}'  # Remove labels for a1 and a2 for now
            for tick in xticks
        ]
        plt.xticks(xticks, labels, fontsize=10)  # Default font size for other ticks
        
        # Add larger, bold custom labels for a1 and a2
        ax.annotate(
            '$\mathit{a}_1$',  # Custom LaTeX label for a1
            xy=(a1, -0.08),  # Position slightly below the x-axis
            xycoords=('data', 'axes fraction'),
            fontsize=15,  # Larger font size for a1
            fontweight='bold',  # Bold font for a1
            ha='center'  # Center-align the label
        )
        ax.annotate(
            '$\mathit{a}_2$',  # Custom LaTeX label for a2
            xy=(a2, -0.08),  # Position slightly below the x-axis
            xycoords=('data', 'axes fraction'),
            fontsize=15,  # Larger font size for a2
            fontweight='bold',  # Bold font for a2
            ha='center'  # Center-align the label
        )

        # Plot legend: 
        plt.legend(fontsize=16,
            bbox_to_anchor=(0.5, 0.5),  # Custom position: (x, y)
        )
        
        #plt.grid(True) # plot grid
        if save_location != False:
            plt.savefig(save_location, dpi=512)
        plt.show()



        
    """
    Make bifurcation diagram for parameter b in black-white
    """
    def make_bif_b(self, b_max = 3, n_vals=2001, save_location=False):

    
            b1 = (2*self.beta/self.alpha - self.a) + 2*np.sqrt((self.beta/self.alpha)*(self.beta/self.alpha - self.a)) # point where 2 more fps emerge
            stab1_x = np.linspace(0, b_max, n_vals)  # stable S=0 branch 
            stab2_x = np.linspace(b1, b_max, n_vals)  # stable top branch
            unstab_x = np.linspace(b1, b_max, n_vals) # unstable branch
            
    
            stab1_y = [0 for i in range(n_vals)] # get y-vals for stable dropout branch
            stab2_y = [ (b_- self.a)/(2*b_) + np.sqrt(self.alpha*(self.alpha*(self.a + b_)**2 - 4*b_*self.beta))/(2*self.alpha*b_) for b_ in stab2_x ] # get y-vals for expertise branch
            unstab_y = [ (b_- self.a)/(2*b_) - np.sqrt(self.alpha*(self.alpha*(self.a + b_)**2 - 4*b_*self.beta))/(2*self.alpha*b_) for b_ in unstab_x] # get y-vals for unstable branch
    
    
            # ----------------
            # --- Plotting ---
            # ----------------
            plt.figure(figsize=(10, 6))
    
            # Plot the 3 branches:
            plt.plot(stab1_x, stab1_y, lw=4, color='black', linestyle='-', label='Stable')
            plt.plot(stab2_x, stab2_y, lw=4, color='black', linestyle='-', label='_nolegend_')
            plt.plot(unstab_x, unstab_y, lw=4, color='black', linestyle=':', label='Unstable')

            # Add vertical lines at b1
            plt.axvline(x=b1, lw=1, color='black', linestyle='--')
            
            # Add arrows showing flow directions
            arrow_gap = 0.15
            arrow_positions_x = [2.5] # Positions for arrows along x-axis
            arrow_positions_y_up =  [- arrow_gap + (b_ - self.a)/(2*b_) - np.sqrt(self.alpha * ((self.alpha * (self.a + b_) ** 2) - 4 *b_* self.beta))/ (2 * self.alpha *b_) for b_ in arrow_positions_x]
            arrow_positions_y_down =  [ arrow_gap + (b_ - self.a)/(2*b_) - np.sqrt(self.alpha * ((self.alpha * (self.a + b_) ** 2) - 4*b_*self.beta))/ (2 * self.alpha *b_) for b_ in arrow_positions_x]
            
    
            for i, x in enumerate(arrow_positions_x):
                # Arrow pointing up (above the unstable branch)
                plt.annotate(
                    '', 
                    xy=(x, arrow_positions_y_up[i]), 
                    xytext=(x, arrow_positions_y_up[i] + arrow_gap),
                    arrowprops=dict(facecolor='black', shrink=0.2, width=1, headwidth=8)
                )
            
                # Arrow pointing down (below the unstable branch)
                plt.annotate(
                    '', 
                    xy=(x, arrow_positions_y_down[i]), 
                    xytext=(x, arrow_positions_y_down[i] - arrow_gap),
                    arrowprops=dict(facecolor='black', shrink=0.2, width=1, headwidth=8)
                )
    
            
            #plt.title('Bifurcation Diagram', fontsize=19)
            plt.xlabel('Parameter $\mathit{b}$', fontsize=19)
            plt.ylim([-0.02, 1])
            plt.xlim(0, b_max)
            plt.ylabel('Skill (S) Fixed Points', fontsize=19)
            
            # Add x-ticks including a1 and a2
            xticks = [0, 1, 3, 4, 5, b1]  # Define ticks
            plt.xticks(xticks)  # Set the tick positions
            
            # Customize all ticks, set labels for numerical ticks
            ax = plt.gca()
            labels = [
                '' if tick == b1 else f'{tick:.1f}'  # Remove labels for b1 for now
                for tick in xticks
            ]
            plt.xticks(xticks, labels, fontsize=10)  # Default font size for other ticks
            
            # Add larger, bold custom labels for b1
            ax.annotate(
                '$\mathit{b}_1$',  # Custom LaTeX label for b1
                xy=(b1, -0.06),  # Position slightly below the x-axis
                xycoords=('data', 'axes fraction'),
                fontsize=15,  # Larger font size for a1
                fontweight='bold',  # Bold font for a1
                ha='center'  # Center-align the label
            )

            #plt.legend(fontsize=16, pos=) # Generate the legend
            plt.legend(fontsize=16,
                bbox_to_anchor=(0.5, 0.5),  # Custom position: (x, y)
            )
            
            #plt.grid(True) # plot grid
            if save_location != False:
                plt.savefig(save_location, dpi=512)
            plt.show()
    


###############################################
### Exponential Model ###
###############################################


class exponential_model:
    """
    * This class defines the dynamical RPS model with logistic learning curve.  
    * It can be used to find fixed points (along with stabilities)
    * It can also plot a bifurcation diagram for any parameter over a range of provided values
    """
    
    def __init__(self, a=0.5, b=2, alpha=0.5, beta=0.3):
        S1, a1, b1, alpha1, beta1 = sp.symbols('S,a,b,alpha,beta') # sympy symbols
        
        # adding symbols RHS of diff. eqn. for dS/dt:
        self.S1 = S1; self.a1 = a1; self.b1 = b1; self.alpha1 = alpha1; self.beta1 = beta1
        self.diff_eqn = (-b1 * alpha1)* S1**2 + ((b1 - a1) * alpha1 - beta1) * S1 + a1*alpha1
        # self.diff_eqn = (-b1 * alpha1) * S1**3 + (b1 - a1) * alpha1 * S1**2 + (a1 * alpha1 - beta1) * S1 
        
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
        out = {'fixed_points': self.fp_vals, 'stability': self.fp_stab, 'derivative': self.fp_der_vals} 
        
        # Note: also returns negative fixed point which is nonsensical 
        
        return out
    
    
    """
    Make bifurcation diagram for parameter a
    """
    def make_bif_a(self, a_max = 3, n_vals=2001, save_location=False):
        
        stab_x = np.linspace(0, a_max, n_vals)
        stab_y = [0 for a_ in stab_x]
        for i in range(len(stab_x)):
            a_ = stab_x[i]
            Discr =  ((a_ + self.b) * self.alpha + self.beta)**2 - 4*self.alpha*self.beta*self.b  # discriminant   
            stab_y[i] = ( ((self.b - a_)*self.alpha - self.beta) + np.sqrt(Discr) )/(2*self.b*self.alpha)


        # ----------------
        # --- Plotting ---
        # ----------------
        plt.figure(figsize=(10, 6))

        # Plot the 3 branches:
        plt.plot(stab_x, stab_y, lw=4, color='black', linestyle='-', label='Stable')
        # plt.plot(unstab_x, unstab_y, lw=4, color='black', linestyle=':', label='Unstable')
        
        # Add arrows showing flow directions
        arrow_gap = 0.15
        arrow_positions_x = [ stab_x[n_vals//2] ] # Positions for arrows along x-axis
        arrow_positions_y_up =  [ stab_y[n_vals//2] for a_ in arrow_positions_x]
        arrow_positions_y_down =  [  stab_y[n_vals//2] for a_ in arrow_positions_x]
        

        for i, x in enumerate(arrow_positions_x):
            # Arrow pointing up (above the unstable branch)
            plt.annotate(
                '', 
                xy=(x, arrow_positions_y_up[i]), 
                xytext=(x, arrow_positions_y_up[i] + arrow_gap ),
                arrowprops=dict(facecolor='black', shrink=0.2, width=1, headwidth=8)
            )
        
            # Arrow pointing down (below the unstable branch)
            plt.annotate(
                '', 
                xy=(x, arrow_positions_y_down[i]), 
                xytext=(x, arrow_positions_y_down[i] - arrow_gap ),
                arrowprops=dict(facecolor='black', shrink=0.2, width=1, headwidth=8)
            )

        
        #plt.title('Bifurcation Diagram', fontsize=19)
        plt.xlabel('Parameter $\mathit{a}$', fontsize=19)
        plt.ylim([-0.02, 1])
        plt.xlim(0, a_max)
        plt.ylabel('Skill (S) Fixed Points', fontsize=19)

        # Plot legend: 
        plt.legend(fontsize=16,
            bbox_to_anchor=(0.95, 0.95),  # Custom position: (x, y)
        )
        
        #plt.grid(True) # plot grid
        if save_location != False:
            plt.savefig(save_location, dpi=512)
        plt.show()

    """
    Make bifurcation diagram for parameter b
    """
    def make_bif_b(self, b_max = 5, n_vals=2001, save_location=False):
        
        stab_x = np.linspace(0, b_max, n_vals)
        stab_y = [0 for b_ in stab_x]
        for i in range(len(stab_x)):
            b_ = stab_x[i]
            Discr =  ((self.a + b_) * self.alpha + self.beta)**2 - 4*self.alpha*self.beta*b_  # discriminant   
            stab_y[i] = ( ((b_ - self.a)*self.alpha - self.beta) + np.sqrt(Discr) )/(2*b_*self.alpha)


        # ----------------
        # --- Plotting ---
        # ----------------
        plt.figure(figsize=(10, 6))

        # Plot the 3 branches:
        plt.plot(stab_x, stab_y, lw=4, color='black', linestyle='-', label='Stable')
        # plt.plot(unstab_x, unstab_y, lw=4, color='black', linestyle=':', label='Unstable')
        
        # Add arrows showing flow directions
        arrow_gap = 0.15
        arrow_positions_x = [ stab_x[n_vals//2] ] # Positions for arrows along x-axis
        arrow_positions_y_up =  [ stab_y[n_vals//2] for b_ in arrow_positions_x]
        arrow_positions_y_down =  [  stab_y[n_vals//2] for b_ in arrow_positions_x]
        

        for i, x in enumerate(arrow_positions_x):
            # Arrow pointing up (above the unstable branch)
            plt.annotate(
                '', 
                xy=(x, arrow_positions_y_up[i]), 
                xytext=(x, arrow_positions_y_up[i] + arrow_gap ),
                arrowprops=dict(facecolor='black', shrink=0.2, width=1, headwidth=8)
            )
        
            # Arrow pointing down (below the unstable branch)
            plt.annotate(
                '', 
                xy=(x, arrow_positions_y_down[i]), 
                xytext=(x, arrow_positions_y_down[i] - arrow_gap ),
                arrowprops=dict(facecolor='black', shrink=0.2, width=1, headwidth=8)
            )

        
        #plt.title('Bifurcation Diagram', fontsize=19)
        plt.xlabel('Parameter $\mathit{b}$', fontsize=19)
        plt.ylim([-0.02, 1])
        plt.xlim(0, b_max)
        plt.ylabel('Skill (S) Fixed Points', fontsize=19)

        # Plot legend: 
        plt.legend(fontsize=16,
            bbox_to_anchor=(0.95, 0.95),  # Custom position: (x, y)
        )
        
        #plt.grid(True) # plot grid
        if save_location != False:
            plt.savefig(save_location, dpi=512)
        plt.show()


 
###############################################
### General Dynamical RPS Learning Function: ###
###############################################
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


 
        
        


