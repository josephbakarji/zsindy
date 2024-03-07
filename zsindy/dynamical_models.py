import numpy as np
import pysindy as ps
import numpy as np
from sympy import symbols, sympify
from itertools import combinations_with_replacement
from collections import Counter
from scipy.integrate import odeint 


###################################################################################


class PolySindyModel:
    """
    A class for defining a polynomial sparse ordinary differential equation.

    Parameters:
    - n_dimensions (int): The number of dimensions in the system.
    - degree (int, optional): The degree of the polynomial terms. Defaults to 3.
    - include_interaction (bool, optional): Whether to include interaction terms. Defaults to True.
    - interaction_only (bool, optional): Whether to include only interaction terms. Defaults to False.
    - include_bias (bool, optional): Whether to include a bias term. Defaults to True.
    """

    def __init__(self, 
                 n_dimensions, 
                 degree=3, 
                 include_interaction=True, 
                 interaction_only=False, 
                 include_bias=True):
        self.n_dimensions = n_dimensions
        self.degree = degree
        self.include_interaction = include_interaction
        self.interaction_only = interaction_only
        self.include_bias = include_bias
        self.combs = list(ps.PolynomialLibrary()._combinations(self.n_dimensions, 
                                                                self.degree, 
                                                                self.include_interaction, 
                                                                self.interaction_only, 
                                                                self.include_bias))

    def symb_library(self, Z): 
        """
        Compute the symbolic polynomial library of terms for the given state vector Z. 

        Parameters:
        - Z (list): The state vector.

        Returns:
        - Zdot_lib (list): The symbolic library of terms.
        """
        Zdot_lib = [] 
        for comb in self.combs:
            term = 1
            for elem in comb:
                term *= Z[elem]
            Zdot_lib.append(term)
        return Zdot_lib
    
    def symb_library_names(self, varnames): 
        """
        Compute the symbolic library of term names for the given variable names.

        Parameters:
        - varnames (list): The variable names.

        Returns:
        - Zdot_lib (list): The symbolic library of term names.
        """
        Zdot_lib = [] 
        for comb in self.combs:
            term = '' 
            for elem in comb:
                term += varnames[elem]
            Zdot_lib.append(term)
        return Zdot_lib

    def poly_system(self, Z, t, Theta):
        """
        Compute the polynomial system of equations.

        Parameters:
        - Z (list): The state vector.
        - t (float): The time.
        - Theta (ndarray): The coefficient matrix.

        Returns:
        - zdot_list (list): The derivative of the state vector.
        """
        zdot_list = []
        for i in range(self.n_dimensions):
            zidot = sum([x*y for x, y in zip(self.symb_library(Z), Theta[i, :])])
            zdot_list.append(zidot)
        return zdot_list



###################################################################################


class DynamicalSystem:
    """
    A class that defines a dynamical system and generates synthetic data to be analyzed with ZSINDy.

    Args:
        equation (function): The equation that describes the dynamics of the system.
        poly_degree (int): The degree of the polynomial library used for modeling.
        args (dict): Additional arguments required by the equation function.
        num_variables (int): The number of variables in the system.

    Attributes:
        equation (function): The equation that describes the dynamics of the system.
        args (dict): Additional arguments required by the equation function.
        varnames (list): The names of the variables in the system.
        library_names (list): The names of the library terms used for modeling.
        true_coefficients (ndarray): The coefficient matrix for the library terms.

    Methods:
        build_coefficients_matrix: Builds the coefficient matrix for the library terms.
        solve: Solves the dynamical system using the given initial conditions and time parameters.
    """

    def __init__(self, equation, poly_degree, args, num_variables):
        self.equation = equation
        self.args = args 
        self.varnames = list('xyzwvqprsu')[:num_variables]
        self.library_names = generate_polynomial_features(poly_degree, self.varnames)
        self.true_coefficients = self.build_coefficients_matrix()

    def build_coefficients_matrix(self):
        """
        Builds the coefficient matrix for the library terms.

        Returns:
            ndarray: The coefficient matrix for the library terms.
        """
        # Define the symbols
        variables = symbols(' '.join(self.varnames))
        
        # Calculate the derivatives using the equation
        derivatives = self.equation(variables, None, self.args)

        # Initialize the library matrix with zeros
        coef_matrix = np.zeros((len(derivatives), len(self.library_names)))

        # Convert library terms from strings to sympy expressions
        lib_terms = [sympify(term, locals={vname: var for vname, var in zip(self.varnames, variables)}) for term in self.library_names]
        
        # For each derivative, find the coefficient of each library term
        for i, derivative in enumerate(derivatives):
            derivative_poly = derivative.as_poly(*variables)
            for j, term in enumerate(lib_terms):
                # Check if the term is in the polynomial, get its coefficient if present
                coeff = derivative_poly.coeff_monomial(term)
                coef_matrix[i, j] = coeff
        
        return coef_matrix

    
    def solve(self, x0, dt, tend):
        """
        Solves the dynamical system using the given initial conditions and time parameters.

        Args:
            x0 (ndarray): The initial conditions of the system.
            dt (float): The time step size.
            tend (float): The end time of the simulation.

        Returns:
            ndarray: The solution of the dynamical system.
        """
        t = np.arange(0, tend, dt)
        self.time = t
        return odeint(lorenz, x0, t)

###################################################################################
    
    
def lorenz(Z, t, params=(10, 28, 8/3)):
    x, y, z = Z
    sigma, rho, beta = params
    return [sigma*(y-x), x*(rho-z)-y, x*y-beta*z]

def rossler(Z, t, params=(0.2, 0.2, 5.7)):
    x, y, z = Z
    a, b, c = params
    return [-y-z, x+a*y, b+z*(x-c)]

def lotka_volterra(Z, t, params=(2/3, 4/3, 1, 1)):
    x, y = Z
    a, b, c, d = params
    return [a*x-b*x*y, c*x*y-d*y]


###################################################################################


def generate_polynomial_features(max_degree, varnames):
    """
    Generate polynomial features up to a given maximum degree.

    Args:
        max_degree (int): The maximum degree of the polynomial features.
        varnames (list): A list of variable names.

    Returns:
        list: A list of polynomial terms up to the given maximum degree.

    Example:
        >>> generate_polynomial_features(2, ['x', 'y'])
        ['1', 'x', 'y', 'x**2', 'x*y', 'y**2']
    """

    # Initialize with a constant term
    terms = ['1']

    # Generate terms for each degree
    for degree in range(1, max_degree + 1):
        for term_powers in combinations_with_replacement(varnames, degree):
            # Count the occurrences of each variable in the combination
            term_counter = Counter(term_powers)

            # Build the term string
            term = '*'.join([f'{var}**{exp}' if exp > 1 else var for var, exp in term_counter.items()])
            terms.append(term)

    return terms

def build_library_matrix(Z, varnames, library_names):
    num_samples, num_features = Z.shape
    assert num_features == len(varnames), "Number of columns in Z must match number of variables."

    # Get the exponent vectors for all terms
    exponent_vectors = [term_to_exponent_vector(term, varnames) for term in library_names]

    # Initialize the library matrix with ones for constant term
    lib_matrix = np.ones((num_samples, len(library_names)))

    # Fill the library matrix
    for j, exponents in enumerate(exponent_vectors):
        for i in range(num_samples):
            # Compute the product of variables raised to their respective exponents
            for var_index, exponent in enumerate(exponents):
                if exponent != 0:  # Skip if the exponent is zero as it does not contribute to the product
                    lib_matrix[i, j] *= Z[i, var_index] ** exponent

    return lib_matrix


#################

def term_to_exponent_vector(term, varnames):
    """
    Converts a term into an exponent vector based on the given variable names.

    Args:
        term (str): The term to convert into an exponent vector.
        varnames (list): A list of variable names.

    Returns:
        tuple: The exponent vector representing the term.

    Example:
        >>> term_to_exponent_vector('x*y**2*z', ['x', 'y', 'z'])
        (1, 2, 1)
    """

    # Initialize a dictionary with all variables set to exponent 0
    exponent_dict = dict.fromkeys(varnames, 0)

    # Split the term into factors accounting for multiplication and power
    factors = term.replace('**', '^').split('*')

    for factor in factors:
        # Handle power terms
        if '^' in factor:
            base, exp = factor.split('^')
            exponent_dict[base] = int(exp)
        # Handle linear terms
        elif factor in varnames:
            exponent_dict[factor] += 1
        # Skip if the factor is '1' as it doesn't change the exponent

    # Convert the dictionary to a tuple using the order of varnames
    exponent_vector = tuple(exponent_dict[var] for var in varnames)

    return exponent_vector


###################################################################################



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from scipy.integrate import odeint as odeint_scipy

    # Simulate
    t = np.linspace(0, 100, 100000, endpoint=False)
    z = odeint_scipy(lorenz, [1, 1, 1], t, args=((10, 28, 8/3),))

    # Discover model with pysindy
    poly_library = ps.PolynomialLibrary(degree=2)
    model = ps.SINDy(feature_library=poly_library,
                    optimizer=ps.STLSQ(threshold=0.05), 
                    feature_names=["x", "y", "z"])

    model.fit(z, t=t)
    model.print()
    lorenz_coefs = model.coefficients()

    psindy = PolySindyModel(n_dimensions=3, degree=2, include_interaction=True, interaction_only=False, include_bias=True)

    x0 = [1, 1, 1]
    delays_dt = 0.3
    dt = 0.006
    tsteps = 10000

    # lorenz coefficients in (10, 3) array
    Theta = lorenz_coefs

    t = np.linspace(0, tsteps*dt, tsteps, endpoint=False)
    zsindy = odeint_scipy(psindy.poly_system, x0, t, args=(lorenz_coefs,))


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(zsindy[:, 0], zsindy[:, 1], zsindy[:, 2])
    plt.show()

##############################################################


