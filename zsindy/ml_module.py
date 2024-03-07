import numpy as np
import matplotlib.pyplot as plt
import pysindy as ps
import pdb
from scipy.integrate import odeint

from itertools import combinations 
from zsindy.dynamical_models import PolySindyModel, generate_polynomial_features, build_library_matrix


#####################################


class ZSindy:
    """
    ZSindy is a class that implements the Sparse Identification of Nonlinear Dynamics (SINDy) algorithm 
    using a Bayesian approach in analogy to statistical mechanics. 'Z' being the partition function.
    This class, defines the input parameters and identifies a parsimonious model, given a time-series input.
    
    Parameters:
    -----------
    rho : float, optional
        The resolution parameter (or variance) for the ZSindy algorithm. Default is 1e-3.
    lmbda : float, optional
        The sparse regularization parameter. Default is 0.
    max_num_terms : int, optional
        The maximum number of non-zero terms to consider on the right hand side of the ODE. Default is 3.
    poly_degree : int, optional
        The degree of the polynomial features to be used. Default is 2.
    variable_names : list of str, optional
        The names of the variables in the system. Default is None.
    use_all_fe : bool, optional
        Whether to use all feature combinations for finding the optimal solution, or use the top combination per |gamma| set size. Default is True.
    sort_all : bool, optional
        Whether to sort feature combinations by the magnitude of the free energy. Default is True.
    normalize_lambda : bool, optional
        Whether to normalize the lambda parameter based on the number of data points. Default is False.

    Methods:
    --------
    fit(x, t)
        Fit the ZSindy model to the given data.
    predict()
        Predict the values of the system variables using the learned model.
    simulate(x0, t, coefs=None)
        Simulate the system using the learned model.
    print()
        Print the learned model equations.
    coefficients()
        Get the learned coefficients of the model.
    set_coefficients(coefs)
        Set the coefficients of the model.
    coefficients_variance()
        Get the variances of the learned coefficients.
    get_free_energies()
        Get the free energies of the feature combinations.
    get_feature_combinations()
        Get the feature combinations used in the model.
    get_gammas()
        Get the indices of the selected features for each variable.
    get_probabilities()
        Get the probabilities of inclusion for each feature combination.
    get_feat_names()
        Get the names of the polynomial features used in the model.
    differentiate_state(x, t)
        Differentiate the state variables with respect to time.
    get_library_matrix(x)
        Get the library matrix for the given data.
    """
    
    def __init__(self, 
                 rho=1e-3, 
                 lmbda=0, 
                 max_num_terms=3, 
                 poly_degree=2,
                 variable_names=None, 
                 use_all_fe=True, 
                 sort_all=True,
                 normalize_lambda=False):

        self.rho = rho
        self.lmbda = lmbda
        self.max_num_terms = max_num_terms
        self.use_all_fe = use_all_fe 
        self.sort_all = sort_all 
        self.variable_names = variable_names
        self.poly_degree = poly_degree
        self.normalize_lambda=normalize_lambda


        # Assuming polynomial (TODO: requires a different class for features in the future) 
        self.sindy_poly_library = ps.PolynomialLibrary(degree=poly_degree)
        self.sindy_model = ps.SINDy(feature_library=self.sindy_poly_library, feature_names=variable_names) 
        

        # Results after call 
        self.Theta = None
        self.prob_opt = None
        self.Fs = None
        self.feat_comb_list = None
        self.zcoef_opt = None
        self.zcoef_opt_vars = None
        self.gamma_opt = None

    def fit(self, x, t):
        """
        Fits the model to the given data.

        Parameters:
        x (numpy.ndarray): The input data.
        t (numpy.ndarray): The time data.

        Returns:
        numpy.ndarray: The optimized coefficients.
        """
        
        # Initialize variables that depend on shape of x
        self.Theta = self.get_library_matrix(x)
        self.num_dims = x.shape[1]
        self.model_integrator = PolySindyModel(n_dimensions=self.num_dims, degree=self.poly_degree)

        xdot = self.differentiate_state(x, t)
        self.prob_opt, self.Fs, self.feat_comb_list, self.zcoef_opt, self.zcoef_opt_vars, self.gamma_opt = \
                self._optimize_all_dimensions(self.Theta, xdot)
        return self.zcoef_opt

    def predict(self):
        """
        Predicts the output using the learned coefficients.

        Raises:
            RuntimeError: If the coefficients have not been computed yet.

        Returns:
            numpy.ndarray: The predicted output.
        """
        if self.Theta is None:
            raise RuntimeError('Returning None. Run fit() first to compute values.')
        return self.Theta @ self.coefficients().T

    def simulate(self, x0, t, coefs=None):
        """
        Simulates the system dynamics using the given initial conditions and time points.

        Parameters:
            x0 (array-like): The initial conditions of the system.
            t (array-like): The time points at which to simulate the system.
            coefs (array-like, optional): The coefficients to use in the simulation. If not provided, the coefficients
                                            obtained from the `coefficients` method will be used.

        Returns:
            array-like: The simulated system dynamics at the given time points.
        """
        if coefs is None:
            coefs = self.coefficients() 
        return odeint(self.model_integrator.poly_system, x0, t, args=(coefs,))

    def print(self):
        feat_names = self.get_feat_names()
        for i in range(self.num_dims):
            print('(', self.variable_names[i], ')\' = ', end="")
            for j in range(self.coefficients().shape[1]):
                if self.coefficients()[i, j] != 0:   
                    print(' + ', end="")
                    print('{:.4f} {:s}'.format(self.coefficients()[i, j], feat_names[j]), end="")
            print("")

    def coefficients(self):
        if self.zcoef_opt is None:
            raise RuntimeError('Returning None. Run fit() first to compute values.')
        return self.zcoef_opt
    
    def set_coefficients(self, coefs):
        self.zcoef_opt = coefs

    def coefficients_variance(self):
        if self.zcoef_opt_vars is None:
            raise RuntimeError('Returning None. Run fit() first to compute values.')
        return self.zcoef_opt_vars

    def get_free_energies(self):
        if self.Fs is None:
            raise RuntimeError('Returning None. Run fit() first to compute values.')
        return self.Fs
    
    def get_feature_combinations(self):
        if self.feat_comb_list is None:
            raise RuntimeError('Returning None. Run fit() first to compute values.')
        return self.feat_comb_list
    
    def get_gammas(self):
        if self.gamma_opt is None:
            raise RuntimeError('Returning None. Run fit() first to compute values.')
        return self.gamma_opt
    
    def get_probabilities(self):
        if self.prob_opt is None:
            raise RuntimeError('Returning None. Run fit() first to compute values.')
        return self.prob_opt

    def get_feat_names(self):
        return generate_polynomial_features(self.poly_degree, self.variable_names)

    def differentiate_state(self, x, t):
        return self.sindy_model.differentiate(x, t=t)

    def get_library_matrix(self, x):
        return build_library_matrix(x, self.variable_names, self.get_feat_names())

    def _optimize_all_dimensions(self, Theta, xdot):
        """
        Optimize the coefficients for all variables.

        Args:
            Theta (numpy.ndarray): The library matrix containing the time-delayed state variables.
            xdot (numpy.ndarray): The time derivatives of the state variables.

        Returns:
            tuple: A tuple containing the following elements:
                - prob_opt (list): The optimal probabilities for each dimension.
                - Fs (list): The normalized objective function values for each dimension.
                - feat_comb_list (list): The feature combinations selected for each dimension.
                - zcoef_opt (numpy.ndarray): The optimal coefficients for each dimension.
                - zcoef_opt_vars (numpy.ndarray): The variances of the optimal coefficients for each dimension.
                - gamma_opt (list): The optimal number of terms selected for each dimension.
        """
        ## Initialize
        Fs = []
        prob_opt = []
        gamma_opt = []
        feat_comb_list = []
        num_features = len(self.get_feat_names())
        zcoef_opt = np.zeros((self.num_dims, num_features))
        zcoef_opt_vars = np.zeros((self.num_dims, num_features))

        ## Loop over dimensions and call _zsindy to find optimal coefficients for each dimension
        for dim in range(self.num_dims):
            prob, Fs_norm, feat_combs, coefs, gammas, coefs_vars = self._optimize_upto_num_terms_per_dim(Theta, xdot, self.max_num_terms, dim)

            # optimal coefficients and their variances
            zcoef_opt[dim, gammas[0]] = coefs[0]
            zcoef_opt_vars[dim, gammas[0]] = coefs_vars[0][gammas[0]]

            # Store values
            Fs.append(Fs_norm)
            feat_comb_list.append(feat_combs)
            prob_opt.append(prob[0])
            gamma_opt.append(gammas[0])

        return prob_opt, Fs, feat_comb_list, zcoef_opt, zcoef_opt_vars, gamma_opt


    def _optimize_upto_num_terms_per_dim(self, Theta, xdot, max_num_terms, dim):
        '''
        Concatenates zsindy for all combinations of features up to max_num_terms, and returns the probability of inclusion of each feature combination,
        the free energy of each feature combination, and the feature combinations sorted by |gamma|.

        Args:
        ------
        Theta (np.array): The library matrix
        xdot (np.array): The time derivative of the data
        dim (int): The index of the variable, such that \dot x_{dim} = f(x)

        Returns:
        --------
        prob (np.array): The probability of inclusion of each feature combination
        Fs_norm (np.array): The free energy of each feature combination
        feat_combs_gmag (list): The list of feature combinations sorted by |gamma|
        coefs_gmag (list): The list of coefficients sorted by |gamma| 
        '''
        Fs_list_gmag = []
        feat_combs_gmag = []
        feat_combs_top = []
        coefs_gmag = []
        coefs_vars_gmag = []
        gammas_all = []

        feat_names_arr = np.array(self.get_feat_names())
        num_features = len(feat_names_arr)

        for i, num_terms in enumerate(range(1, max_num_terms+1)):
            Fs_list, coef_list, coef_vars_list, gammas  = self._optimize_per_num_terms_per_dim(Theta, xdot, num_terms, dim)

            # sorted per |gamma| only
            sorted_idx = np.argsort(Fs_list)
            if self.normalize_lambda:
                lmbda_norm = normalize_lambda_numdata(self.lmbda, self.rho, xdot.shape[0])
            else:
                lmbda_norm = self.lmbda
            Fs_list = self._sparse_penalty_fe(Fs_list[sorted_idx], lmbda_norm, num_terms) 
            sorted_gammas = gammas[sorted_idx]

            feat_combs_gmag += [', '.join(comb) for comb in feat_names_arr[sorted_gammas]]
            feat_combs_top += [', '.join(feat_names_arr[sorted_gammas[0]])]

            Fs_list_gmag.append(Fs_list)
            coefs_gmag.append(coef_list[sorted_idx])
            coefs_vars_gmag.append(coef_vars_list[sorted_idx])
            gammas_all.append(sorted_gammas)

        
        Fs_all = np.concatenate(Fs_list_gmag)
        Fs_norm = Fs_all - np.min(Fs_all) 
        feat_combs = feat_combs_gmag       
        coefs = coefs_gmag
        coefs_vars = coefs_vars_gmag

        if self.sort_all:
            Fs_sort_idx = np.argsort(Fs_norm)
            Fs_norm = Fs_norm[Fs_sort_idx]
            feat_combs = [feat_combs_gmag[i] for i in Fs_sort_idx]
            coefs = [sum([list(coef) for coef in coefs_gmag], [])[i] for i in Fs_sort_idx]
            coefs_vars = [sum([list(coef) for coef in coefs_vars_gmag], [])[i] for i in Fs_sort_idx]
            gammas_all = [sum([list(gam) for gam in gammas_all], [])[i] for i in Fs_sort_idx]

        prob = np.exp(-Fs_norm)/np.sum(np.exp(-Fs_norm))

        return prob, Fs_norm, feat_combs, coefs, gammas_all, coefs_vars




    def _optimize_per_num_terms_per_dim(self, Theta, xdot, num_terms, dim):
        '''
        Returns the optimal coefficients for the given data and model parameters for a single equation

        Args:
            Theta (np.array): The feature matrix
            xdot (np.array): The derivative of the data

        Returns:
            Fs_list (np.array): The free energy of each feature combination
            coef_list (np.array): The optimal coefficients for each feature combination
            gammas (np.array): The list of feature combinations
        '''

        C = Theta.T @ Theta 
        V = Theta.T @ xdot
        num_features = len(self.get_feat_names())
        gammas, Fs, mean_coefs = self._free_energy_coefs(C, V, self.rho, num_terms, num_features, dim)
        Fs_list, coef_list = self._dict_to_lists(Fs, mean_coefs, gammas)
        coef_var_list = [self.rho**2 * np.diag(np.linalg.inv(C)) for gamma in gammas]

        return np.array(Fs_list), np.array(coef_list), np.array(coef_var_list), np.array(gammas)
    

    def _sparse_penalty_fe(self, Fs, lmbda, num_terms):
        """
        Applies a sparse penalty to the given feature matrix.

        Parameters:
        - Fs (numpy.ndarray): The feature matrix.
        - lmbda (float): The sparse penalty parameter.
        - num_terms (int): The number of terms in the penalty.

        Returns:
        - numpy.ndarray: The feature matrix with the sparse penalty applied.
        """
        return Fs + lmbda * num_terms

    def _normalized_fe(self, Fs):
        """
        Normalize the given feature vector Fs by subtracting the minimum value.

        Parameters:
        Fs (numpy.ndarray): The feature vector to be normalized.

        Returns:
        numpy.ndarray: The normalized feature vector.
        """
        return Fs - np.min(Fs)

    def _compute_partition(self, Fs, lmbda, num_terms):
        """
        Compute the partition function for a given set of features.

        Parameters:
        - Fs (numpy.ndarray): The input features.
        - lmbda (float): The penalty parameter.
        - num_terms (int): The number of terms to consider.

        Returns:
        - float: The computed partition function.
        """

        Fs_norm = self._sparse_penalty_fe(Fs, lmbda, num_terms)
        Fs_norm = self._normalized_fe(Fs_norm)
        return np.sum(np.exp(-Fs_norm))

    def _zprobability(self, Fs):
        """
        Calculates the probability distribution of a given set of free energies.

        Parameters:
        - Fs (numpy.ndarray): Array of free energies.

        Returns:
        - numpy.ndarray: Array of probabilities corresponding to the input free energies.
        """
        return np.exp(-Fs)/np.sum(np.exp(-Fs))

    def _free_energy(self, C, V, rho, gamma):
        """
        Calculate the free energy.

        Parameters:
        - C (numpy.ndarray): Covariance matrix.
        - V (numpy.ndarray): Data matrix.
        - rho (float): Resolution (distribution standard deviation) parameter.
        - gamma (list): List of indices of library terms to be included.

        Returns:
        - float: The calculated free energy.
        """
        subC = C[gamma][:, gamma]
        subV = V[gamma]
        tempF = - len(gamma)*0.5*np.log(2*np.pi*rho**2)+0.5*np.linalg.slogdet(subC)[1] \
            - 0.5/rho**2 * (subV @ np.linalg.inv(subC) @ subV)
        return tempF


    def _free_energy_coefs(self, C, V, rho, num_terms, num_feats, dim):
        """
        Calculate the free energy coefficients for a given set of inputs.

        Parameters:
        - C (numpy.ndarray): Covariance matrix.
        - V (numpy.ndarray): Data matrix.
        - rho (float): The resolution parameter.
        - num_terms (int): The number of terms.
        - num_feats (int): The number of features.
        - dim (int): The variable index to consider.

        Returns:
        - gammas (dict): A dictionary containing the index combinations.
        - Fs (dict): A dictionary containing the free energy values for each index combination.
        - mean_coefs (dict): A dictionary containing the mean coefficients for each index combination.
        """
        gammas = self._get_idx_combinations(num_feats, num_terms)
        Fs = {key: None for key in gammas}
        mean_coefs = {key: None for key in gammas}
        for i, gamma in enumerate(gammas):
            lgamma = list(gamma)
            Fs[gamma] = self._free_energy(C, V[:, dim], rho, lgamma)
            mean_coefs[gamma] = np.linalg.inv(C[lgamma][:, lgamma]) @ V[lgamma, dim]

        return gammas, Fs, mean_coefs


    def _get_idx_combinations(self, list_len, num_terms):
        '''
        Returns all possible combinations of length num_terms from a list of length list_len
        '''
        return [i for i in combinations(range(list_len), num_terms)]

    
    # ## Not used?
    # def _get_name_combinations_per_num_terms(self, num_terms, feat_names):
    #     '''
    #     Returns all possible name combinations of length num_terms from a list of features of length list_len
    #     '''
    #     combs = self._get_idx_combinations(len(feat_names), num_terms)
    #     feat_comb_names = []
    #     for comb in combs:
    #         feat_name = []
    #         for term_idx in comb:
    #             feat_name.append(feat_names[term_idx]+', ')
    #         feat_comb_names.append(''.join(feat_name)[:-2]) 
    #     return  feat_comb_names

    # def _get_name_combinations_upto_num_terms(self, max_num_terms, feat_names):
    #     '''
    #     Returns all possible name combinations of length up to max_num_terms from a list of features of length list_len 
    #     '''
    #     feat_comb_names = []
    #     for num_terms in range(max_num_terms+1):
    #         feat_comb_names += self._get_name_combinations_per_num_terms(num_terms, feat_names)
    #     return feat_comb_names

    def _dict_to_lists(self, Fs, mean_coefs, gammas):
        Fs_list = []
        coef_list = []
        for g in gammas:
            Fs_list.append(Fs[g])
            coef_list.append(mean_coefs[g])
        return Fs_list, coef_list



#############################
#############################
#############################

def normalize_lambda(Lambda, rho, t, dt):
    return Lambda * t/dt * rho**2

def normalize_lambda_numdata(Lambda, rho, N):
    return Lambda * N * rho**2