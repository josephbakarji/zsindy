import pysindy as ps
import numpy as np
from zsindy.ml_module import ZSindy 
from zsindy.dynamical_models import lorenz, DynamicalSystem
import matplotlib.pyplot as plt 
from itertools import product
from tqdm import tqdm

class PhaseDiagram:
    """
    Class representing a phase diagram.

    Parameters:
    -----------
    X : numpy.ndarray
        The data matrix.
    t : numpy.ndarray
        The time vector.
    inputs : dict
        Dictionary containing inputs to vary for the phase diagram.
    zsindy_num_terms : int
        The maximum number of terms in the ZSindy model.
    poly_degree : int
        The degree of the polynomial library.
    varnames : list
        The names of the variables.
    x_variable : str
        The key of the x-axis input.
    y_variable : str
        The key of the y-axis input.
    get_accuracy : bool, optional
        Flag to indicate if accuracy should be calculated.
    tidx_pred_horizon : int, optional
        Time index horizon.
    time_div_threshold : float, optional
        Threshold for time division.
    true_feats : dict, optional
        The true feature combinations.
    """

    def __init__(self, X, t, inputs, zsindy_num_terms, poly_degree, varnames, x_variable, y_variable, get_accuracy=False, tidx_pred_horizon=2000, time_div_threshold=0.01, true_feats=None):
        self.X = X
        self.t = t
        self.inputs = inputs
        self.zsindy_num_terms = zsindy_num_terms
        self.x_variable = x_variable 
        self.y_variable = y_variable 
        self.poly_degree = poly_degree
        self.true_feats = true_feats
        self.varnames = varnames
        self.get_accuracy = get_accuracy
        self.tidx_pred_horizon=tidx_pred_horizon
        self.time_div_threshold=time_div_threshold
        self.index_cache = {}  # Cache for storing index combinations

    def index_combinations(self):
        """
        Generate all possible index combinations for the inputs.

        Returns:
        --------
        combinations : list
            List of all possible index combinations.
        sorted_keys : list
            List of sorted input keys.
        """
        sorted_keys = sorted(self.inputs.keys())
        list_lengths = [range(len(self.inputs[key])) for key in sorted_keys]
        combinations = list(product(*list_lengths))
        return combinations, sorted_keys

    def _find_tidx_div(self, x, y, threshold):
        """
        Find the time index where the difference between two arrays exceeds a threshold.

        Parameters:
        -----------
        x : numpy.ndarray
            First array.
        y : numpy.ndarray
            Second array.
        threshold : float
            Threshold value.

        Returns:
        --------
        tidx_div : int
            Time index where the difference exceeds the threshold.
        """
        for i in range(len(x)):
            if np.abs(x[i] - y[i]) > threshold:
                return i
        return len(x)

    def setup_sindy_model(self):
        """
        Set up the SINDy model.

        Returns:
        --------
        model : SINDy
            The SINDy model.
        poly_library : PolynomialLibrary
            The polynomial library.
        """
        poly_library = ps.PolynomialLibrary(degree=self.poly_degree)
        model = ps.SINDy(feature_library=poly_library, optimizer=ps.STLSQ(threshold=self.sindy_thresh))
        return model, poly_library

    def add_noise_to_data(self, X, eta):
        """
        Add noise to the data.

        Parameters:
        -----------
        X : numpy.ndarray
            The data matrix.
        eta : float
            The noise level.

        Returns:
        --------
        X_noisy : numpy.ndarray
            The data matrix with added noise.
        """
        return X + eta * np.random.randn(*X.shape)

    def sweep_params(self):
        """
        Combined method for generating phase diagrams.

        Parameters:
        -----------
        inputs : dict
            Dictionary containing inputs to vary for the phase diagram.
        true_feats : dict, optional
            The true feature combinations.
        dims : list, optional
            The list of indices of the variables.
        x : str
            The key of the x-axis input.
        y : str
            The key of the y-axis input.
        tidx_hor : int
            Time index horizon.
        time_div_threshold : float
            Threshold for time division.
        full_phase : bool
            Flag to indicate if full phase diagram is needed.

        Returns:
        --------
        Depends on the flag 'full_phase'. It could be probabilities, ranks, etc.
        """

        # sindy_model, poly_library = self.setup_sindy_model()

        input_idx_combs, sorted_keys = self.index_combinations()

        probabilities = dict.fromkeys(input_idx_combs)
        Fs_norms = dict.fromkeys(input_idx_combs)
        feats = dict.fromkeys(input_idx_combs)
        coefs = dict.fromkeys(input_idx_combs)
        accuracy = None
        div_time = None
        if self.get_accuracy:
            accuracy = dict.fromkeys(input_idx_combs)
            div_time = dict.fromkeys(input_idx_combs)

        eta_idx = sorted_keys.index('eta')
        lmbda_idx = sorted_keys.index('lmbda')
        dataratio_idx = sorted_keys.index('dataratio')

        if 'rho' in self.inputs.keys():
            rho_idx = sorted_keys.index('rho')

        for input_element in tqdm(input_idx_combs):

            eta = self.inputs['eta'][input_element[eta_idx]]
            lmbda = self.inputs['lmbda'][input_element[lmbda_idx]]
            dataratio = self.inputs['dataratio'][input_element[dataratio_idx]]
            if 'rho' in self.inputs.keys():
                rho = self.inputs['rho'][input_element[rho_idx]]
            else:
                rho = eta

            idx_end = int(np.floor(dataratio*len(self.t)-1))
            t_short = self.t[:idx_end]

            # Add noise to data
            Xnoisy = self.X[:idx_end, :] + eta * np.random.randn(*self.X[:idx_end, :].shape)

            zsindy_model = ZSindy(rho=rho, lmbda=lmbda, max_num_terms=self.zsindy_num_terms, 
                            poly_degree=self.poly_degree, variable_names=self.varnames)
            
            # Fit zsindy
            zcoef_opt = zsindy_model.fit(Xnoisy, t_short)
            prob_opt = zsindy_model.get_probabilities()
            Fs = zsindy_model.get_free_energies()
            feat_comb_list = zsindy_model.get_feature_combinations()

            probabilities[input_element] = prob_opt
            Fs_norms[input_element] = Fs
            feats[input_element] = feat_comb_list
            coefs[input_element] = zcoef_opt

            if self.get_accuracy:

                zX_pred = zsindy_model.simulate(self.X[0], t=t_short[:self.tidx_pred_horizon])
                diff = np.nan_to_num(zX_pred - self.X[:self.tidx_pred_horizon, :])
                accuracy[input_element] = np.linalg.norm(diff, 2) 
                div_time[input_element] = [self._find_tidx_div(zX_pred[:, dim], self.X[:self.tidx_pred_horizon, dim], self.time_div_threshold) for dim in range(zX_pred.shape[1])]

        results = {}
        results['probabilities'] = probabilities
        results['Fs_norms'] = Fs_norms
        results['feats'] = feats 
        results['coefs'] = coefs

        if self.get_accuracy:
            results['div_time'] = div_time 
            results['accuracy'] = accuracy 

        return results



    def build_diag_accuracy(self, results):
        """
        Builds and returns the accuracy algorithmic phase diagram.

        Parameters:
        - results (dict): A dictionary containing the results of the experiment.

        Returns:
        - true_acc_flipped (ndarray): A 2D numpy array for the accuracy matrix (flipped to be visualized with imshow).
        - true_predtime_flipped (ndarray): A 2D numpy array for prediction time span.
        - xtix (list): A list of strings representing the x-axis tick labels.
        - ytix (list): A list of strings representing the y-axis tick labels.
        """
        # Method implementation...
    def build_diag_accuracy(self, results):
        
        feats = results['feats']
        accuracy = results['accuracy']
        div_time = results['div_time']

        x = self.x_variable
        y = self.y_variable
        inputs = self.inputs

        input_idx_combs, sorted_keys = self.index_combinations()
        input_comb_idxes = feats.keys()

        eta_idx = sorted_keys.index(y)
        lmbda_idx = sorted_keys.index(x)

        true_acc = np.zeros((len(inputs[y]), len(inputs[x])))
        true_predtime = np.zeros((len(inputs[y]), len(inputs[x])))

        for incomb_idx in input_comb_idxes:
            true_acc[incomb_idx[eta_idx], incomb_idx[lmbda_idx]] = accuracy[incomb_idx] # rank of true features 
            true_predtime[incomb_idx[eta_idx], incomb_idx[lmbda_idx]] = np.mean(div_time[incomb_idx])
            
        xtix = ["{:.1e}".format(lmbda) for lmbda in inputs[x]]
        ytix = ["{:.1e}".format(eta) for eta in inputs[y][::-1]] # reverse order for eta

        true_acc_flipped = true_acc[::-1, :]
        true_predtime_flipped = true_predtime[::-1, :]

        return true_acc_flipped, true_predtime_flipped, xtix, ytix 


    def build_diag_setsize(self, results):
        """
        Build the phase diagram for set size gamma based on the results dictionary.

        Args:
            results (dict): A dictionary containing the results of the experiment, with keys: 'feats', 'coefs'.

        Returns:
            tuple: A tuple containing the following:
                - set_size_flipped (ndarray): The set size matrix, where each element represents the number of terms in ODE.
                - xtix (list): The x-axis tick labels.
                - ytix (list): The y-axis tick labels.
        """

        feats = results['feats']
        coefs = results['coefs']

        x = self.x_variable
        y = self.y_variable
        inputs = self.inputs

        input_idx_combs, sorted_keys = self.index_combinations()
        input_comb_idxes = feats.keys()

        eta_idx = sorted_keys.index(y)
        lmbda_idx = sorted_keys.index(x)

        set_size = np.zeros((len(inputs[y]), len(inputs[x])))

        for incomb_idx in input_comb_idxes:
            num_non_zero_terms = np.count_nonzero(coefs[incomb_idx])
            set_size[incomb_idx[eta_idx], incomb_idx[lmbda_idx]] = num_non_zero_terms

        xtix = ["{:.1e}".format(lmbda) for lmbda in inputs[x]]
        ytix = ["{:.1e}".format(eta) for eta in inputs[y][::-1]]  # reverse order for eta

        set_size_flipped = set_size[::-1, :]

        return set_size_flipped, xtix, ytix


    def build_diag_rank_prob(self, results):
        """
        Builds and returns the true probability and rank matrices for each dimension.

        Args:
            results (dict): A dictionary containing the results of the experiment.
                - 'feats': The features used in the experiment.
                - 'probabilities': The probabilities of each feature combination.
                - 'Fs_norms': The normalized free energies.
                - 'feats_result': The features used in the experiment.

        Returns:
            tuple: A tuple containing the following elements:
                - true_prob_flipped_dim (list): A list of true probability matrices for each dimension.
                - true_rank_flipped_dim (list): A list of true rank matrices for each dimension.
                - xtix (list): A list of x-axis tick labels.
                - ytix (list): A list of y-axis tick labels.
        """
        feats = results['feats']
        probabilities_result = results['probabilities']
        Fs_norms_result = results['Fs_norms']
        feats_result = results['feats']

        x = self.x_variable
        y = self.y_variable
        inputs = self.inputs

        input_idx_combs, sorted_keys = self.index_combinations()
        input_comb_idxes = feats.keys()

        eta_idx = sorted_keys.index(y)
        lmbda_idx = sorted_keys.index(x)

        true_prob_dim = []
        true_rank_dim = []

        print(self.X.shape[1])
        for dim in range(self.X.shape[1]):
            true_prob = np.zeros((len(inputs[y]), len(inputs[x])))
            true_rank = np.zeros((len(inputs[y]), len(inputs[x])))

            for incomb_idx in input_comb_idxes:
                probabilities = {key: val[dim] for key, val in probabilities_result.items()}
                Fs_norms = {key: val[dim] for key, val in Fs_norms_result.items()} 
                feats = {key: val[dim] for key, val in feats_result.items()} 
                Fs_sorted_idx = np.argsort(Fs_norms[incomb_idx])

                true_comb_idx = feats[incomb_idx].index(self.true_feats[dim]) # true combination index in sorted free energies
                true_prob[incomb_idx[eta_idx], incomb_idx[lmbda_idx]] = probabilities[incomb_idx] # probabilities
                true_rank[incomb_idx[eta_idx], incomb_idx[lmbda_idx]] = np.where(true_comb_idx == Fs_sorted_idx)[0][0] # rank of true features 

            true_prob_dim.append(true_prob)
            true_rank_dim.append(true_rank)

        xtix = ["{:.1e}".format(lmbda) for lmbda in inputs[x]]
        ytix = ["{:.1e}".format(eta) for eta in inputs[y][::-1]] # reverse order for eta

        true_prob_flipped_dim = [true_prob_dim[i][::-1, :] for i in range(len(true_prob_dim))]
        true_rank_flipped_dim = [true_rank_dim[i][::-1, :] for i in range(len(true_rank_dim))]

        return true_prob_flipped_dim, true_rank_flipped_dim, xtix, ytix




    def plot_phase_per_dim(self, prob, rank, xtix, ytix, cmap='viridis',
                           x_label=None, y_label=None, title='', figsize=(8, 10),
                           label_size=10, tix_size=5):
        """
        Plot the phase per dimension.

        Parameters:
        - prob (list): List of 2D arrays representing the probability of inclusion for each combination.
        - rank (list): List of 2D arrays representing the rank of true combinations for each dimension.
        - xtix (list): List of x-axis tick labels.
        - ytix (list): List of y-axis tick labels.
        - cmap (str, optional): Colormap to use for the plots. Defaults to 'viridis'.
        - x_label (str, optional): Label for the x-axis. Defaults to None.
        - y_label (str, optional): Label for the y-axis. Defaults to None.
        - title (str, optional): Title for the plot. Defaults to an empty string.
        - figsize (tuple, optional): Figure size. Defaults to (8, 10).
        - label_size (int, optional): Font size for labels. Defaults to 10.
        - tix_size (int, optional): Font size for tick labels. Defaults to 5.
        """
        inputs = self.inputs
        xvar = self.x_variable
        yvar = self.y_variable

        fig = plt.figure(figsize=figsize) 
        fig.suptitle(title, fontsize=12)
        ax = fig.subplots(len(prob), 2)
        dims = range(len(prob))

        for i in range(len(dims)):
            ax00 = ax[i, 0].imshow(prob[i], cmap=cmap)
            fig.colorbar(ax00, ax=ax[i, 0], location='right', anchor=(0, 0.3), shrink=0.9)
            ax[i, 0].set_xlabel(x_label, fontsize=label_size)
            ax[i, 0].set_ylabel(y_label, fontsize=label_size)
            ax[i, 0].set_xticks(range(len(inputs[xvar])), xtix, rotation=90, fontsize=tix_size)
            ax[i, 0].set_yticks(range(len(inputs[yvar])), ytix, fontsize=tix_size)
            ax[i, 0].set_title('Inc. Prob. of True Comb. dim='+ str(i), fontsize=label_size)

            ax01 = ax[i, 1].imshow(rank[i], cmap=cmap)
            fig.colorbar(ax01, ax=ax[i, 1], location='right', anchor=(0, 0.3), shrink=0.9)
            ax[i, 1].set_xlabel(x_label, fontsize=label_size)
            ax[i, 1].set_ylabel(y_label, fontsize=label_size)
            ax[i, 1].set_xticks(range(len(inputs[xvar])), xtix, rotation=90, fontsize=tix_size)
            ax[i, 1].set_yticks(range(len(inputs[yvar])), ytix, fontsize=tix_size)
            ax[i, 1].set_title('Rank of True Comb. dim=' + str(i), fontsize=label_size)

        plt.tight_layout()
        



    def plot_phase(self, true_acc, true_predtime, xtix, ytix, x_label=None, title='',
                   y_label=None, write_values=False, cmap='hot'):
        """
        Plot the phase diagram for prediction accuracy and time horizon predictability.

        Parameters:
        - true_acc (numpy.ndarray): Array of prediction accuracy values.
        - true_predtime (numpy.ndarray): Array of time horizon predictability values.
        - xtix (list): List of x-axis tick labels.
        - ytix (list): List of y-axis tick labels.
        - x_label (str, optional): Label for the x-axis. Defaults to None.
        - title (str, optional): Title for the plot. Defaults to an empty string.
        - y_label (str, optional): Label for the y-axis. Defaults to None.
        - write_values (bool, optional): Whether to write the accuracy values in each square. Defaults to False.
        - cmap (str, optional): Colormap for the plot. Defaults to 'hot'.

        Returns:
        - None
        """

        inputs = self.inputs
        xvar = self.x_variable
        yvar = self.y_variable
        x_label = xvar if x_label is None else x_label
        y_label = yvar if y_label is None else y_label

        fig = plt.figure(figsize=(10, 5)) 
        # title
        fig.suptitle(title, fontsize=12)
        ax1 = fig.add_subplot(121) 
        # Write the accuracy in each square 
        if write_values:
            for j in range(len(inputs[xvar])): 
                for k in range(len(inputs[yvar])): 
                    text = ax1.text(j, k, np.round(true_acc[k, j], 4), ha="center", va="center", color="w", fontsize=7)
        ax1.imshow(np.log(true_acc), cmap=cmap)
        # fig.colorbar(ax01, ax=ax1, location='right', anchor=(0, 0.3), shrink=0.3)
        ax1.set_xlabel(x_label)
        ax1.set_ylabel(y_label)
        ax1.set_title('Prediction Accuracy')
        ax1.set_xticks(range(len(inputs[xvar])), xtix, rotation=90, fontsize=5)
        ax1.set_yticks(range(len(inputs[yvar])), ytix, fontsize=5)

        ax2 = fig.add_subplot(122) 
        # Write the accuracy in each square 
        if write_values:
            for j in range(len(inputs[xvar])): 
                for k in range(len(inputs[yvar])): 
                    text = ax2.text(j, k, np.round(true_predtime[k, j], 4), ha="center", va="center", color="w", fontsize=7)
        ax2.imshow(np.log(true_predtime), cmap=cmap)
        # fig.colorbar(ax02, ax=ax2, location='right', anchor=(0, 0.3), shrink=0.3)
        ax2.set_xlabel(x_label)
        ax2.set_ylabel(y_label)
        ax2.set_title('Time Horizon Predictability')
        ax2.set_xticks(range(len(inputs[xvar])), xtix, rotation=90, fontsize=5)
        ax2.set_yticks(range(len(inputs[yvar])), ytix, fontsize=5)

        plt.tight_layout()



    def plot_phase_setsize(self, setsize, xtix, ytix, x_label=None, title='',
                           y_label=None, write_values=False, cmap='hot'):
        """
        Plot the phase diagram of the set size.

        Parameters:
        - setsize (numpy.ndarray): The phase set size array.
        - xtix (list): The x-axis tick labels.
        - ytix (list): The y-axis tick labels.
        - x_label (str, optional): The label for the x-axis. Defaults to None.
        - title (str, optional): The title of the plot. Defaults to an empty string.
        - y_label (str, optional): The label for the y-axis. Defaults to None.
        - write_values (bool, optional): Whether to write the values in each square. Defaults to False.
        - cmap (str, optional): The colormap to use. Defaults to 'hot'.
        """

        inputs = self.inputs
        xvar = self.x_variable
        yvar = self.y_variable
        x_label = xvar if x_label is None else x_label
        y_label = yvar if y_label is None else y_label

        fig = plt.figure(figsize=(10, 5)) 
        ax1 = fig.add_subplot(111) 

        if write_values:
            for j in range(len(inputs[xvar])): 
                for k in range(len(inputs[yvar])): 
                    text = ax1.text(j, k, setsize[k, j], ha="center", va="center", color="w", fontsize=7)
        
        ax1.imshow(setsize, cmap=cmap)
        ax1.set_xlabel(x_label)
        ax1.set_ylabel(y_label)
        ax1.set_title('Number of Nonzero Terms')
        ax1.set_xticks(range(len(inputs[xvar])), xtix, rotation=90, fontsize=5)
        ax1.set_yticks(range(len(inputs[yvar])), ytix, fontsize=5)

        plt.tight_layout()



if __name__ == '__main__':


    # Zsindy parameters
    rho = 0.1
    zsindy_num_terms = 4
    lmbda = 1e8
    poly_degree = 2

    eta = 0.01

    ## Simulate dynamical system
    x0 = (0, 1, 1.05)
    tend = 20
    dt = 0.005
    args = (10, 28, 8/3)

    dyn = DynamicalSystem(lorenz, poly_degree, args, num_variables=len(x0))
    X = dyn.solve(x0, dt, tend)
    true_coefs = dyn.true_coefficients
    varnames = dyn.varnames
    lambda_norm = dyn.time/dt * rho**2

    # Generate those in DynamicalSystem class
    # NOTE: using '*' here, while not using it in previous code
    true_feats = ['x, y', 'x, y, x*z', 'z, x*y']  
    num_dims = X.shape[1]

    # Add Noise
    X += eta * np.random.randn(*X.shape)
    t = dyn.time

    ## Z-Sindy
    zmodel = ZSindy(rho=rho, 
                    lmbda=lmbda, 
                    max_num_terms=zsindy_num_terms, 
                    poly_degree=poly_degree,
                    variable_names=varnames)

    zmodel.fit(X, dyn.time)

    print("\nZ-Sindy Model:")
    zmodel.print()
    z_x_pred = zmodel.simulate(x0, dyn.time)
    z_xdot_pred = zmodel.predict()

    ####################

    dims = [0, 1, 2]
    sindy_degree = 2
    sindy_thresh = 0.01
    max_num_terms = 3
    x_variable = 'lmbda'
    y_variable = 'eta'
    get_accuracy = True
    cmap = 'viridis'

    x_label=r'$\lambda$, Sparsity parameter'
    y_label=r'$\eta$, Noise magniture'
    write_values=True

    rhos = np.array([1e-6, 1e-4, 1e-2, 1])

    inputs = {'eta': np.array([1e-10, 1e-3, 1e-2, 1]),
            'lmbda': np.array([0, 1e4, 1e8, 1e12]),
            'dataratio': np.array([1.0])}

    diag = PhaseDiagram(X, t, inputs, zsindy_num_terms, 
                               poly_degree, varnames, x_variable, 
                               y_variable, get_accuracy=get_accuracy, true_feats=true_feats) 
    results = diag.sweep_params()
    true_acc, true_predtime, xtix, ytix = diag.build_diag_accuracy(results)
    diag.plot_phase(true_acc, true_predtime, xtix, ytix, x_label=x_label, y_label=y_label, cmap=cmap) 

    prob, rank, xtix, ytix = diag.build_diag_rank_prob(results)
    diag.plot_phase_per_dim(prob, rank, xtix, ytix, x_label=x_label, y_label=y_label, cmap=cmap) 
    
    plt.show()