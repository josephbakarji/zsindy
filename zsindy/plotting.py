import matplotlib.pyplot as plt
import numpy as np
# import gaussian kde
from scipy.stats import gaussian_kde


def gaussian(x, mu, sig):
    return 1.0 / (np.sqrt(2.0 * np.pi) * sig) * np.exp(-np.power((x - mu) / sig, 2.0) / 2)
    
class EvsZSindyPlotter:
    """
    A class for plotting results comparing E-SINDy and Z-SINDy.

    Args:
        true_coefs (ndarray): The true coefficients.
        feat_names (list): The names of the features.
        varnames (list): The names of the variables.

    Methods:
        e_vs_zsindy_coefs: Plots the coefficients of E-SINDy and Z-SINDy.
        e_vs_zsindy_coef_error: Plots the absolute error of E-SINDy and Z-SINDy coefficients.
        e_vs_zsindy_simulations: Plots the simulations of E-SINDy and Z-SINDy.
        e_sindy_alltrials: Plots the ensemble of E-SINDy predictions.
        e_vs_zsindy_coefficient_distributions: Plots the coefficient distributions of E-SINDy and Z-SINDy.
        e_vs_zsindy_simulation_envelopes: Plots the mean time series and standard deviation envelopes of E-SINDy and Z-SINDy predictions.
    """
    
    def __init__(self, true_coefs, feat_names, varnames):
        self.true_coefs = true_coefs
        self.feat_names = feat_names
        self.varnames = varnames

    def e_vs_zsindy_coefs(self, e_mean, e_std, z_mean, z_std, z_scale, figsize=(13, 3)):
        num_dims = e_mean.shape[0]
        num_feats = e_mean.shape[1]

        fig, axes = plt.subplots(1, num_dims, figsize=figsize)
        for i, ax in enumerate(axes):
            ax.errorbar(np.arange(num_feats), e_mean[i, :], yerr=e_std[i, :], capsize=4, fmt='o', label='E-Sindy')
            ax.errorbar(np.arange(num_feats), z_mean[i, :], yerr=z_scale*z_std[i, :], capsize=4, fmt='^', label='Z-Sindy')
            ax.plot(np.arange(num_feats), self.true_coefs[i, :], 'x', label='True')
            ax.set_xticks(np.arange(num_feats))
            ax.set_xticklabels(['$'+term+'$' for term in self.feat_names], rotation=90)
            ax.set_xlabel('Term')
            if i == 0:
                ax.set_ylabel('Coefficient value')
            ax.set_title(f'${self.varnames[i]}$')
            ax.grid(True)
            ax.legend()

        plt.tight_layout()
        return fig, axes

    def e_vs_zsindy_coef_error(self, e_mean, z_mean, figsize=(13, 3)):
        num_dims = e_mean.shape[0]
        num_feats = e_mean.shape[1]
        zsindy_coef_error = np.sqrt((z_mean - self.true_coefs)**2)
        esindy_coef_error = np.sqrt((e_mean - self.true_coefs)**2)

        fig = plt.figure(figsize=figsize)
        for i in range(num_dims):
            plt.subplot(1, num_dims, i+1)
            plt.scatter(np.arange(num_feats), esindy_coef_error[i, :], marker='o', label='E-Sindy')
            plt.scatter(np.arange(num_feats), zsindy_coef_error[i, :], marker='^', label='Z-Sindy')
            plt.xticks(np.arange(num_feats), ['$'+term+'$' for term in self.feat_names], rotation=90)
            plt.xlabel('Term')
            if i == 0:
                plt.ylabel('Absolute Error')
            plt.title(f'{self.varnames[i]}')
            plt.grid()
            plt.legend()


    def e_vs_zsindy_simulations(self, xdot, xpred, e_xdot, e_xpred, z_xdot, z_xpred, t, figsize=(12, 7)):
        num_dims = e_xdot.shape[1]
        fig = plt.figure(figsize=figsize)
        for i in range(num_dims):
            plt.subplot(num_dims, 1, i+1)
            plt.plot(t, xdot[:, i], label='Data')
            plt.plot(t, e_xdot[:, i], label='E-Sindy')
            plt.plot(t, z_xdot[:, i], label='Z-Sindy')
            plt.xlabel('Time')
            plt.ylabel(f'$d/dt {self.varnames[i]}$')
            plt.legend()

        fig = plt.figure(figsize=figsize)
        for i in range(num_dims):
            plt.subplot(num_dims, 1, i+1)
            plt.plot(t, xpred[:, i], label='Data')
            plt.plot(t, z_xpred[:, i], label='Z-Sindy')
            plt.plot(t, e_xpred[:, i], label='E-Sindy')
            plt.xlabel('Time')
            plt.ylabel(f'${self.varnames[i]}$')
            plt.legend()


    def e_sindy_alltrials(self, xpred, e_xpred_ensemble, t, figsize=(10, 7)):
        num_dims = xpred.shape[1]

        fig = plt.figure(figsize=figsize)
        for i in range(num_dims):
            plt.subplot(num_dims, 1, i+1)
            plt.plot(t, xpred[:, i], label='Data', lw=1, alpha=.5)
            for j in range(e_xpred_ensemble.shape[0]):
                plt.plot(t, e_xpred_ensemble[j, :, i], lw=.1)
            plt.xlabel('Time')
            plt.ylabel(self.varnames[i])

        plt.tight_layout() 

    def e_vs_zsindy_coefficient_distributions(self, e_coef_ensemble, z_mean, z_std, z_scale, varnames, feat_names, bins=10, figsize=(15, 5)):
        num_dims = e_coef_ensemble.shape[1]
        num_feats = e_coef_ensemble.shape[2]
        z_std_scaled = z_scale*z_std

        fig, axes = plt.subplots(num_dims, num_feats, figsize=figsize)
        for i in range(num_dims):
            for j in range(num_feats):

                # This means that the distribution is only plotted for non-zero z_std_scaled (e_sindy is not shown)
                five_std = np.max([z_std_scaled[i, j], np.std(e_coef_ensemble[:, i, j])]) * 5
                x_range = np.linspace(z_mean[i, j] - five_std, z_mean[i, j] + five_std, 100)

                # estimate the distribution of coefficients using kernel density estimation and plot
                # kde of e-sindy coefficients

                if np.all(e_coef_ensemble[:, i, j] == 0):
                    kernel = gaussian_kde(e_coef_ensemble[:, i, j] + 1e-5*np.random.normal(0, size=e_coef_ensemble.shape[0]))
                else:
                    kernel = gaussian_kde(e_coef_ensemble[:, i, j])
                e_pdf = kernel(x_range)
                axes[i, j].plot(x_range, e_pdf, label='E-Sindy')

                z_pdf = 0
                if z_std_scaled[i, j] > 0:
                    z_pdf = gaussian(x_range, z_mean[i, j], z_std_scaled[i, j])
                    axes[i, j].plot(x_range, z_pdf, label='Z-Sindy')

                if j == 0:
                    axes[i, j].set_ylabel(f'PDF - ${varnames[i]}$')
                if i == len(varnames)-1:
                    axes[i, j].set_xlabel(f'${feat_names[j]}$')
                
                axes[i, j].plot([self.true_coefs[i, j], self.true_coefs[i, j]], [0, max([np.max(z_pdf), np.max(e_pdf)])], 'k--', label='True')
                
                # axes[i, j].set_xticks([])
                axes[i, j].tick_params(axis='x', labelsize=8)
                axes[i, j].set_yticks([])
                # axes[i, j].set_title('Coefficient distributions for ' + str(self.varnames[i]))
                if i == 0 and j == 0:
                    axes[i, j].legend()

        plt.tight_layout()
        plt.subplots_adjust(wspace=0.2, hspace=0.2)


    def e_vs_zsindy_simulation_envelopes(self, e_xpred_ensemble, z_xpred_ensemble, time, varnames):
        """
        Plot the mean time series and the standard deviation envelope for both
        E-SINDy and Z-SINDy model predictions.

        :param time: The time vector for the x-axis.
        :param e_xpred_ensemble: Ensemble of time series predictions from E-SINDy.
        :param z_xpred_ensemble: Ensemble of time series predictions from Z-SINDy.
        :param varnames: Names of the variables for the legend.
        """
        # Calculate mean and standard deviation for E-SINDy
        e_mean_pred = np.mean(e_xpred_ensemble, axis=0)
        e_std_pred = np.std(e_xpred_ensemble, axis=0)

        # Calculate mean and standard deviation for Z-SINDy
        z_mean_pred = np.mean(z_xpred_ensemble, axis=0)
        z_std_pred = np.std(z_xpred_ensemble, axis=0)

        # Plotting
        plt.figure(figsize=(14, 5))
        
        # Loop through each dimension/variable
        for i, var in enumerate(varnames):
            plt.subplot(len(varnames), 1, i+1)
            
            # E-SINDy plot with std envelope
            plt.plot(time, e_mean_pred[:, i], label=f'E-SINDy Mean', color='b')
            plt.fill_between(time, 
                            e_mean_pred[:, i] - 2*e_std_pred[:, i], 
                            e_mean_pred[:, i] + 2*e_std_pred[:, i], 
                            color='blue', alpha=0.2)
            
            # Z-SINDy plot with std envelope
            plt.plot(time, z_mean_pred[:, i], label=f'Z-SINDy Mean', color='r')
            plt.fill_between(time, 
                            z_mean_pred[:, i] - 2*z_std_pred[:, i], 
                            z_mean_pred[:, i] + 2*z_std_pred[:, i], 
                            color='red', alpha=0.2)
            
            plt.xlabel(f'Time, $t$')
            plt.ylabel(f'${var}$')
            plt.legend()
        
        plt.tight_layout()