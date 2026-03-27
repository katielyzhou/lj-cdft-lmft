import numpy as np
import LJ_lmft_utils as lmft
import plot_utils as plt
import neural_utils as neural
import LJEOS as eos

alpha_updates_default_onetype = {
    10: 0.0001,
    20: 0.001,
    50: 0.001,
    100: 0.005,
    300: 0.005,
    900: 0.008,
    2000: 0.01,
    5000: 0.1,
}

alpha_updates_default_onetype_LR = {
    10: 0.0001,
    20: 0.0005,
    50: 0.0006,
    100: 0.0008,
    300: 0.001,
    900: 0.002,
    2000: 0.005,
    2500: 0.007,
    3000: 0.01,
    3500: 0.05,
    4000: 0.1,
}
alpha_updates_default_twotype = {
    10: 0.000001,
    20: 0.00001,
    50: 0.00002,
    100: 0.00004,
    300: 0.00005,
    500: 0.00006,
    1000: 0.0001,
    1200: 0.0002,
    1400: 0.0003,
    1500: 0.0004,
    1600: 0.0005,
    1700: 0.0006,
    1800: 0.0007,
    1900: 0.0008,
    2100: 0.0009,
    2200: 0.001,
    2300: 0.0011,
    2400: 0.0012,
    2600: 0.0013,
    2800: 0.0014,
    3000: 0.0015,
    3200: 0.0016,
    3400: 0.0018,
    3600: 0.0020,
    3800: 0.0021,
    4000: 0.0023,
    4200: 0.0025,
    4400: 0.0026,
    4600: 0.0028,
    4800: 0.0030,
    5000: 0.0032,
    5200: 0.0035,
    5400: 0.0036,
    5500: 0.0040,
    5800: 0.0050,
    6000: 0.0060,
    6200: 0.008,
    6400: 0.015,
    7000: 0.02,
    8000: 0.03,
    9000: 0.07,
    10000: 0.08,
}


def minimise_SR(model, zbins, muloc, T, initial_guess, input_bins=1001,
                        plot=False, maxiter=10000, 
                        alpha_initial=1e-6, alpha_updates=None, 
                        print_every=1000, plot_every=1000, tolerance=5e-6,
                        output_dict=False):
    """
    Calculate the density profile with neural DFT using a standard Picard iteration.

    Parameters:
    - model (tf.keras.Model): Keras model for calculating the one-body direct correlation function.
    - zbins (array-like): Spatial grid points.
    - muloc (array-like): Local chemical potential
    - T (float): Temperature
    - plot (bool): Toggle for interactive plotting.
    - maxiter (int): Maximum number of Picard steps.
    - alpha_initial (float): Initial value for the Picard parameter alpha.
    - alpha_updates (dict): Iteration thresholds and corresponding alpha values.
    - initial_guess (float): Initial guess for the density profile.
    - print_every (int): Print the iteration number every n steps.
    - plot_every (int): Update the plot every n steps.
    - tolerance (float): Convergence tolerance.

    Returns:
    - tuple: z coordinates and density profile.
    """
    
    # setting up grid
    rho_new = np.zeros_like(zbins)
    valid = np.isfinite(muloc)
    rho = initial_guess * np.ones_like(zbins)
    L = zbins[-1] - zbins[0]
    log_rho_new = np.zeros_like(zbins)
    log_rho = np.zeros_like(zbins)
    valid = np.isfinite(muloc)
    log_rho[valid] = np.log(initial_guess)
    log_rho[~valid] = -np.inf 

    # Picard iteration parameter
    alpha = alpha_initial
    if alpha_updates is None:
        alpha_updates = alpha_updates_default_onetype

    if plot:
        fig, ax = plt.configure_plot(zbins)
        color_count = 0


    for i in range(maxiter + 1):
        if i in alpha_updates:
            alpha = alpha_updates[i]

        if plot and i % plot_every == 0:
            plt.plot_interactive_SR_onetype(fig, ax, zbins, rho, muloc, color_count)
            color_count += 1

        # correlation from trained SR model
        c1_pred = neural.c1_onetype_T(model, rho, T, input_bins, output_dict=output_dict)
        
        # update density
        log_rho_new[valid] = muloc[valid] + c1_pred[valid]
        log_rho_new[~valid] = -np.inf 
        rho_new = np.exp(log_rho_new)
        log_rho = (1 - alpha) * log_rho + alpha * log_rho_new
        rho = np.exp(log_rho)

        # check convergence
        delta = np.max(np.abs(rho_new - rho))

        if np.isnan(delta):
            print("Not converged: delta is NaN")
            return None, None
        
        relative_error = delta / np.max(rho)
        
        if plot and i % print_every == 0:
            print(f"Iteration {i}: delta = {delta}")

        if delta < tolerance or relative_error < tolerance:
            print(f"Converged after {i} iterations (delta = {delta})")
            if plot:
                plt.plot_end_SR_onetype(zbins, rho, muloc, ax)
            return zbins, rho

    print(f"Not converged after {i} iterations (delta = {delta})")
    return None, None


def minimise_LR(model, zbins, Vext, T, eps=1, sig=1, rc=2.5, rho_bulk=None, mu = None,
                        initial_guess=0.5, input_bins=1001, plot=True, maxiter=10000, 
                        alpha_initial=1e-6, alpha_updates=None, 
                        print_every=1000, plot_every=100, tolerance=5e-6):
    """
    Calculate the density profile with neural DFT using a standard Picard iteration for a long range system.

    Parameters:
    - model (tf.keras.Model): Keras model for calculating the one-body direct correlation function.
    - zbins (array-like): Spatial grid points.
    - Vext (array-like): Negative of the External potential in units of kbT (-beta Vext)
    - rho_bulk: bulk density (leave mu unspecified)
    - mu: chemical potential of long range system (leave rho_bulk unspecified) in units of kbT (beta mu)
    - plot (bool): Toggle for interactive plotting.
    - maxiter (int): Maximum number of Picard steps.
    - alpha_initial (float): Initial value for the Picard parameter alpha.
    - alpha_updates (dict): Iteration thresholds and corresponding alpha values.
    - initial_guess (float): Initial guess for the density profile.
    - print_every (int): Print the iteration number every n steps.
    - plot_every (int): Update the plot every n steps.
    - tolerance (float): Convergence tolerance.

    Returns:
    - tuple: z coordinates and density profile.
    """
    if (mu is None and rho_bulk is None) or (mu is not None and rho_bulk is not None):
        raise ValueError("Specify either the chemical potential 'mu' or the mean density 'rho_bulk'")
    
    T = T * np.ones_like(zbins)

    # setting up grid
    rho_new = np.zeros_like(zbins)
    valid = np.isfinite(Vext)
    rho = initial_guess * np.ones_like(zbins)
    L = zbins[-1] - zbins[0]
    log_rho_new = np.zeros_like(zbins)
    log_rho = np.zeros_like(zbins)
    valid = np.isfinite(Vext)
    log_rho[valid] = np.log(initial_guess)
    log_rho[~valid] = -np.inf 
    beta = 1/T # Assuming kb=1
    bin_width = abs(zbins[1] - zbins[0]) 

    potential_array = lmft.attract(zbins, eps=eps, sig=sig, cutoff=rc)

    # Calculate delta mu
    if rho_bulk is not None:
        rho_b = rho_bulk
        mu_LR = eos.calc_mu(rho_bulk, T[0], sig=sig, rc=rc)

    if mu is not None:
        mu_LR = mu*T[0]
        rho_b = eos.calc_rhob(mu, T[0], eps=eps, sig=sig, rc=rc)

    mu_R = np.log(rho_b) - np.mean(neural.c1_onetype_T(model, rho_b*np.ones_like(T), T, input_bins)) # beta mu
    mu_correction = mu_LR - mu_R*T[0]

    # Picard iteration parameter
    alpha = alpha_initial
    if alpha_updates is None:
        alpha_updates = alpha_updates_default_onetype_LR

    if plot:
        fig, ax = plt.configure_plot(zbins)
        color_count = 0

    for i in range(maxiter + 1):
        if i in alpha_updates:
            alpha = alpha_updates[i]

        if plot and i % plot_every == 0:
            plt.plot_interactive_LR_onetype_LJ(fig, ax, zbins, rho, Vext, color_count)
            color_count += 1

        # correlation from trained SR model
        c1_pred_SR = neural.c1_onetype_T(model, rho, T, input_bins)

        V_correction = lmft.V_correction(bin_width, rho-rho_b, potential_array)

        c1_LR = c1_pred_SR + beta*V_correction - beta*mu_correction

        # update density
        log_rho_new[valid] = Vext[valid] + beta[valid]*mu_LR + c1_LR[valid] 
        log_rho_new[~valid] = -np.inf 
        rho_new = np.exp(log_rho_new)
        log_rho = (1 - alpha) * log_rho + alpha * log_rho_new
        rho = np.exp(log_rho)
        
        # check convergence
        delta = np.max(np.abs(rho_new - rho))

        if np.isnan(delta):
            print("Not converged: delta is NaN")
            return None, None
        
        relative_error = delta / np.max(rho)
        
        if plot and i % print_every == 0:
            print(f"Iteration {i}: delta = {delta}")

        if delta < tolerance or relative_error < tolerance:
            print(f"Converged after {i} iterations (delta = {delta})")
            if plot:
                plt.plot_end_LR_onetype_LJ(zbins, rho, Vext+beta*mu_LR, ax)
            return zbins, rho

    print(f"Not converged after {i} iterations (delta = {delta})")
    return None, None



def minimise_LR_twotype_onemodel(model, zbins, T,
                        Vext_A, Vext_B,
                        eps_AA=1, eps_BB=1, interaction_parameter=0,
                        sigma_AA=1, sigma_BB=1,
                        rc_AA = 2.5, rc_AB = 2.5, rc_BB = 2.5, shift = True,
                        rho_bulk_A = None, mu_A = None,
                        rho_bulk_B = None, mu_B = None,
                        initial_guess_A=0.04, initial_guess_B=0.04,
                        input_bins=1201,
                        plot=True, maxiter=100000, alpha_initial=0.000001, 
                        alpha_updates=None,
                        print_every=1000, plot_every=1000, tolerance=1e-5,
                        output_dict=False):
    """
    Calculate the density profile with neural DFT using a standard Picard iteration 
    for two types of particles for long-range interactions.

    Parameters:
    - model (tf.keras.Model): The Keras model to be used for the calculation of the 
                              one-body direct correlation function.
    - zbins (array-like): Spatial grid points.
    - Vext (array-like): Negative of the External potential in units of kbT (-beta Vext)
    - rho_bulk: bulk density to calculate mu_R for delta mu
    - mu: chemical potential of long range system in units of kbT (beta mu)
    - interaction_parameter: diagonal element of array, where eps_ij = (1-interaction_parameter)*sqrt(eps_ii*eps_jj)
    - shift: True for truncated and shifted; False for truncated
    - plot (bool): Toggle interactive plotting.
    - maxiter (int): Maximum number of Picard steps.
    - alpha_initial (float): Initial value for the relaxation parameter alpha.
    - alpha_updates (dict): Dictionary of iteration thresholds and corresponding 
                            alpha values to update alpha during iterations.
    - initial_guess (float): Initial guess for the density profile.

    Returns:
    - tuple: z coordinates and density profile.
    """

    T = T * np.ones_like(zbins)
    L = zbins[-1] - zbins[0]
    beta = 1/T # Assuming kb=1
    bin_width = abs(zbins[1] - zbins[0])

    eps_AB = (1-interaction_parameter) * np.sqrt(eps_AA*eps_BB)
    sigma_AB = 0.5 * (sigma_AA + sigma_BB)
    
    # setting up grid
    rho_A_new = np.zeros_like(zbins)
    rho_B_new = np.zeros_like(zbins)
    validA = np.isfinite(Vext_A) 
    validB = np.isfinite(Vext_B)
    rho_A = initial_guess_A * np.ones_like(zbins)
    rho_B = initial_guess_B * np.ones_like(zbins)
    log_rho_A_new = np.zeros_like(zbins)
    log_rho_B_new = np.zeros_like(zbins)
    log_rho_A = np.zeros_like(zbins)
    log_rho_B = np.zeros_like(zbins)
    log_rho_A[validA] = np.log(initial_guess_A)
    log_rho_B[validB] = np.log(initial_guess_B)
    log_rho_A[~validA] = -np.inf
    log_rho_B[~validB] = -np.inf
    
    potential_array_AA = lmft.attract(zbins, eps=eps_AA, sig=sigma_AA, cutoff=rc_AA, shift = shift)
    potential_array_AB = lmft.attract(zbins, eps=eps_AB, sig=sigma_AB, cutoff=rc_AB, shift = shift)
    potential_array_BB = lmft.attract(zbins, eps=eps_BB, sig=sigma_BB, cutoff=rc_BB, shift = shift)

    # Calculate delta mu
  
    mu_LR_A = mu_A*T[0]
    mu_LR_B = mu_B*T[0]

    if rho_bulk_A is None and rho_bulk_B is None:
         # may give wrong densities when near phase coexistence
         rho_b_a, rho_b_b = eos.calc_rhob_mixture(np.array([mu_LR_A, mu_LR_B]), T[0], np.array([eps_AA, eps_BB]),
                                               np.array([sigma_AA, sigma_BB]), np.array([[0, interaction_parameter], [interaction_parameter, 0]]))
    else:
        rho_b_a = rho_bulk_A
        rho_b_b = rho_bulk_B

    rho_tot = rho_b_a + rho_b_b

    mu_R_A = np.log(rho_b_a) - np.mean(neural.c1_onetype_T(model, rho_tot*np.ones_like(T), T, input_bins)) # beta mu
    mu_correction_A = mu_LR_A - mu_R_A*T[0]
    mu_R_B = np.log(rho_b_b) - np.mean(neural.c1_onetype_T(model, rho_tot*np.ones_like(T), T, input_bins)) # beta mu
    mu_correction_B = mu_LR_B - mu_R_B*T[0]

    # Picard iteration parameter
    alpha = alpha_initial
    if alpha_updates is None:
        alpha_updates = alpha_updates_default_twotype
    
    if plot:
        fig, ax = plt.configure_plot(zbins)
        color_count = 0
  
    for i in range(maxiter + 1):
        if i in alpha_updates:
            alpha = alpha_updates[i]
        
        if plot and i % plot_every == 0:
            plt.plot_interactive_SR_twotype(fig, ax, zbins, rho_A, rho_B, Vext_A+beta*mu_LR_A, Vext_B+beta*mu_LR_B, color_count)
            color_count += 1
            
        # correlation from trained SR model
        c1_pred_SR = neural.c1_onetype_T(model, rho_A+rho_B, T, input_bins)
        
        V_correction_AA = lmft.V_correction(bin_width, rho_A - rho_b_a, potential_array_AA)
        V_correction_AB = lmft.V_correction(bin_width, rho_B - rho_b_b, potential_array_AB)
        V_correction_BA = lmft.V_correction(bin_width, rho_A - rho_b_a, potential_array_AB)
        V_correction_BB = lmft.V_correction(bin_width, rho_B - rho_b_b, potential_array_BB)

        
        c1_LR_A = c1_pred_SR + beta*(V_correction_AA + V_correction_AB) - beta*mu_correction_A
        c1_LR_B = c1_pred_SR + beta*(V_correction_BB + V_correction_BA) - beta*mu_correction_B

        # update density
        log_rho_A_new[validA] = Vext_A[validA] + beta[validA]*mu_LR_A + c1_LR_A[validA]
        log_rho_B_new[validB] = Vext_B[validB] + beta[validB]*mu_LR_B + c1_LR_B[validB]
        log_rho_A_new[~validA] = -np.inf
        log_rho_B_new[~validB] = -np.inf
        rho_A_new = np.exp(log_rho_A_new)
        rho_B_new = np.exp(log_rho_B_new)
        log_rho_A = (1 - alpha) * log_rho_A + alpha * log_rho_A_new
        log_rho_B = (1 - alpha) * log_rho_B + alpha * log_rho_B_new


        rho_A = np.exp(log_rho_A)
        rho_B = np.exp(log_rho_B)
    
        
        delta_A = np.max(np.abs(rho_A_new - rho_A))
        delta_B = np.max(np.abs(rho_B_new - rho_B))
        delta = max(delta_A, delta_B)
        
        if np.isnan(delta):
            print("Not converged: delta is NaN")
            return  None, None, None

        relative_error = delta / max(np.max(rho_B), np.max(rho_A))
        
        if i % print_every == 0:
            print(f"Iteration {i}: delta = {delta}")

        if delta < tolerance or relative_error < tolerance:
            print(f"Converged after {i} iterations (delta = {delta})")
            if plot:
                plt.plot_end_SR_twotype(zbins, rho_A, rho_B, Vext_A+beta*mu_LR_A, Vext_B+beta*mu_LR_B, ax)
            return zbins, rho_A, rho_B
        
    print(f"Not converged after {maxiter} iterations (delta = {delta})")
    return None, None, None






def minimise_LR_parameters(model, zbins, Vext, parameters, eps=1, sig=1, rc=2.5, rho_bulk=None, mu = None,
                        initial_guess=0.5, input_bins=1001, plot=True, maxiter=10000, 
                        alpha_initial=1e-6, alpha_updates=None, 
                        print_every=1000, plot_every=100, tolerance=5e-6):
    """
    Calculate the density profile with neural DFT using a standard Picard iteration for a long range system.

    Parameters:
    - model (tf.keras.Model): Keras model for calculating the one-body direct correlation function.
    - zbins (array-like): Spatial grid points.
    - Vext (array-like): Negative of the External potential in units of kbT (-beta Vext)
    - parameters (dict): Parameters for the model - must contain the temperature.
    - rho_bulk: bulk density (leave mu unspecified)
    - mu: chemical potential of long range system (leave rho_bulk unspecified) in units of kbT (beta mu)
    - plot (bool): Toggle for interactive plotting.
    - maxiter (int): Maximum number of Picard steps.
    - alpha_initial (float): Initial value for the Picard parameter alpha.
    - alpha_updates (dict): Iteration thresholds and corresponding alpha values.
    - initial_guess (float): Initial guess for the density profile.
    - print_every (int): Print the iteration number every n steps.
    - plot_every (int): Update the plot every n steps.
    - tolerance (float): Convergence tolerance.

    Returns:
    - tuple: z coordinates and density profile.
    """
    if (mu is None and rho_bulk is None) or (mu is not None and rho_bulk is not None):
        raise ValueError("Specify either the chemical potential 'mu' or the mean density 'rho_bulk'")

    T = parameters["T"]
    T = T * np.ones_like(zbins)

    # setting up grid
    rho_new = np.zeros_like(zbins)
    valid = np.isfinite(Vext)
    rho = initial_guess * np.ones_like(zbins)
    L = zbins[-1] - zbins[0]
    log_rho_new = np.zeros_like(zbins)
    log_rho = np.zeros_like(zbins)
    valid = np.isfinite(Vext)
    log_rho[valid] = np.log(initial_guess)
    log_rho[~valid] = -np.inf 
    beta = 1/T # Assuming kb=1
    bin_width = abs(zbins[1] - zbins[0]) 

    potential_array = lmft.attract(zbins, eps=eps, sig=sig, cutoff=rc)
    # Calculate delta mu
    if rho_bulk is not None:
        rho_b = rho_bulk
        mu_LR = eos.calc_mu(rho_bulk, T[0], sig=sig, rc=rc)

    if mu is not None:
        mu_LR = mu*T[0]
        rho_b = eos.calc_rhob(mu, T[0], eps=eps, sig=sig, rc=rc)
            
    mu_R = np.log(rho_b) - np.mean(neural.c1_onetype_params(model, rho_b*np.ones_like(rho), parameters, input_bins)) # beta mu
    mu_correction = mu_LR - mu_R*T[0]

    # Picard iteration parameter
    alpha = alpha_initial
    if alpha_updates is None:
        alpha_updates = alpha_updates_default_onetype_LR

    if plot:
        fig, ax = plt.configure_plot(zbins)
        color_count = 0

    for i in range(maxiter + 1):
        if i in alpha_updates:
            alpha = alpha_updates[i]

        if plot and i % plot_every == 0:
            plt.plot_interactive_LR_onetype_LJ(fig, ax, zbins, rho, Vext, color_count)
            color_count += 1

        # correlation from trained SR model
        c1_pred_SR = neural.c1_onetype_params(model, rho, parameters, input_bins)

        V_correction = lmft.V_correction(bin_width, rho-rho_b, potential_array)

        c1_LR = c1_pred_SR + beta*V_correction - beta*mu_correction

        # update density
        log_rho_new[valid] = Vext[valid] + beta[valid]*mu_LR + c1_LR[valid] 
        log_rho_new[~valid] = -np.inf 
        rho_new = np.exp(log_rho_new)
        log_rho = (1 - alpha) * log_rho + alpha * log_rho_new
        rho = np.exp(log_rho)
        
        # check convergence
        delta = np.max(np.abs(rho_new - rho))

        if np.isnan(delta):
            print("Not converged: delta is NaN")
            return None, None
        
        relative_error = delta / np.max(rho)
        
        if plot and i % print_every == 0:
            print(f"Iteration {i}: delta = {delta}")

        if delta < tolerance or relative_error < tolerance:
            print(f"Converged after {i} iterations (delta = {delta})")
            if plot:
                plt.plot_end_LR_onetype_LJ(zbins, rho, Vext+beta*mu_LR, ax)
            return zbins, rho

    print(f"Not converged after {i} iterations (delta = {delta})")
    return None, None



######### Extras #########

def minimise_LR_twotype_MF(model, zbins, T,
                        Vext_A, Vext_B, symmetric=False,
                        eps_AA=1, eps_BB=1, interaction_parameter=0,
                        sigma_AA=1, sigma_BB=1,
                        rc_AA = 2.5, rc_AB = 2.5, rc_BB = 2.5, shift = True,
                        rho_bulk_A = None, mu_A = None,
                        rho_bulk_B = None, mu_B = None,
                        initial_guess_A=0.04, initial_guess_B=0.04,
                        input_bins=1201,
                        plot=True, maxiter=100000, alpha_initial=0.000001, 
                        alpha_updates=None,
                        print_every=1000, plot_every=1000, tolerance=1e-5,
                        output_dict=False):
    """
    Calculate the density profile with neural DFT using a standard Picard iteration 
    for two types of particles for long-range interactions.

    Parameters:
    - model (tf.keras.Model): The Keras model to be used for the calculation of the 
                              one-body direct correlation function.
    - zbins (array-like): Spatial grid points.
    - Vext (array-like): Negative of the External potential in units of kbT (-beta Vext)
    - rho_bulk: bulk density (leave mu unspecified)
    - mu: chemical potential of long range system in units of kbT (beta mu)
    - interaction_parameter: diagonal element of array, where eps_ij = (1-interaction_parameter)*sqrt(eps_ii*eps_jj)
    - plot (bool): Toggle interactive plotting.
    - maxiter (int): Maximum number of Picard steps.
    - alpha_initial (float): Initial value for the relaxation parameter alpha.
    - alpha_updates (dict): Dictionary of iteration thresholds and corresponding 
                            alpha values to update alpha during iterations.
    - initial_guess (float): Initial guess for the density profile.

    Returns:
    - tuple: z coordinates and density profile.
    """
    """
    if (mu_A is None and rho_bulk_A is None) or (mu_A is not None and rho_bulk_A is not None):
        raise ValueError("Specify either the chemical potential 'mu' or the mean density 'rho_bulk' for species A")
    
    if (mu_B is None and rho_bulk_B is None) or (mu_B is not None and rho_bulk_B is not None):
        raise ValueError("Specify either the chemical potential 'mu' or the mean density 'rho_bulk' for species B")
    """
    T = T * np.ones_like(zbins)
    L = zbins[-1] - zbins[0]
    beta = 1/T # Assuming kb=1
    bin_width = abs(zbins[1] - zbins[0])

    eps_AB = (1-interaction_parameter) * np.sqrt(eps_AA*eps_BB)
    sigma_AB = 0.5 * (sigma_AA + sigma_BB)
    
    # setting up grid
    rho_A_new = np.zeros_like(zbins)
    rho_B_new = np.zeros_like(zbins)
    validA = np.isfinite(Vext_A) 
    validB = np.isfinite(Vext_B)
    rho_A = initial_guess_A * np.ones_like(zbins)
    rho_B = initial_guess_B * np.ones_like(zbins)
    log_rho_A_new = np.zeros_like(zbins)
    log_rho_B_new = np.zeros_like(zbins)
    log_rho_A = np.zeros_like(zbins)
    log_rho_B = np.zeros_like(zbins)
    log_rho_A[validA] = np.log(initial_guess_A)
    log_rho_B[validB] = np.log(initial_guess_B)
    log_rho_A[~validA] = -np.inf
    log_rho_B[~validB] = -np.inf
    
    potential_array_AA = lmft.attract(zbins, eps=eps_AA, sig=sigma_AA, cutoff=rc_AA, shift = shift)
    potential_array_AB = lmft.attract(zbins, eps=eps_AB, sig=sigma_AB, cutoff=rc_AB, shift = shift)
    potential_array_BB = lmft.attract(zbins, eps=eps_BB, sig=sigma_BB, cutoff=rc_BB, shift = shift)

    mu_LR_A = mu_A*T[0]
    mu_LR_B = mu_B*T[0]

    if rho_bulk_A is None and rho_bulk_B is None:
         # may give wrong densities when near phase coexistence
         rho_b_a, rho_b_b = eos.calc_rhob_mixture(np.array([mu_LR_A, mu_LR_B]), T[0], np.array([eps_AA, eps_BB]),
                                               np.array([sigma_AA, sigma_BB]), np.array([[0, interaction_parameter], [interaction_parameter, 0]]))
    else:
        rho_b_a = rho_bulk_A
        rho_b_b = rho_bulk_B


    # Picard iteration parameter
    alpha = alpha_initial
    if alpha_updates is None:
        alpha_updates = alpha_updates_default_twotype
    
    if plot:
        fig, ax = plt.configure_plot(zbins)
        color_count = 0
  
    for i in range(maxiter + 1):
        if i in alpha_updates:
            alpha = alpha_updates[i]
        
        if plot and i % plot_every == 0:
            plt.plot_interactive_SR_twotype(fig, ax, zbins, rho_A, rho_B, Vext_A+beta*mu_LR_A, Vext_B+beta*mu_LR_B, color_count)
            color_count += 1
            
        # correlation from trained SR model
        c1_pred_SR = neural.c1_onetype_T(model, rho_A+rho_B, T, input_bins)
        
        V_correction_AA = lmft.V_correction(bin_width, rho_A, potential_array_AA)
        V_correction_AB = lmft.V_correction(bin_width, rho_B, potential_array_AB)
        V_correction_BA = lmft.V_correction(bin_width, rho_A, potential_array_AB)
        V_correction_BB = lmft.V_correction(bin_width, rho_B, potential_array_BB)

        
        c1_LR_A = c1_pred_SR + beta*(V_correction_AA + V_correction_AB)
        c1_LR_B = c1_pred_SR + beta*(V_correction_BB + V_correction_BA)

        # update density
        log_rho_A_new[validA] = Vext_A[validA] + beta[validA]*mu_LR_A + c1_LR_A[validA]
        log_rho_B_new[validB] = Vext_B[validB] + beta[validB]*mu_LR_B + c1_LR_B[validB]
        log_rho_A_new[~validA] = -np.inf
        log_rho_B_new[~validB] = -np.inf
        rho_A_new = np.exp(log_rho_A_new)
        rho_B_new = np.exp(log_rho_B_new)
        log_rho_A = (1 - alpha) * log_rho_A + alpha * log_rho_A_new
        log_rho_B = (1 - alpha) * log_rho_B + alpha * log_rho_B_new

        if symmetric == True:
            # Enforce reflection symmetry about center
            log_rho_A = 0.5 * (log_rho_A + log_rho_A[::-1])
            log_rho_B = 0.5 * (log_rho_B + log_rho_B[::-1])

        rho_A = np.exp(log_rho_A)
        rho_B = np.exp(log_rho_B)
    
        
        delta_A = np.max(np.abs(rho_A_new - rho_A))
        delta_B = np.max(np.abs(rho_B_new - rho_B))
        delta = max(delta_A, delta_B)
        
        if np.isnan(delta):
            print("Not converged: delta is NaN")
            return  None, None, None

        relative_error = delta / max(np.max(rho_B), np.max(rho_A))
        
        if i % print_every == 0:
            print(f"Iteration {i}: delta = {delta}")

        if delta < tolerance or relative_error < tolerance:
            print(f"Converged after {i} iterations (delta = {delta})")
            if plot:
                plt.plot_end_SR_twotype(zbins, rho_A, rho_B, Vext_A+beta*mu_LR_A, Vext_B+beta*mu_LR_B, ax) 
            return zbins, rho_A, rho_B
        
    print(f"Not converged after {maxiter} iterations (delta = {delta})")

    return None, None, None #zbins, best_rho_A, best_rho_B


def minimise_LR_twotype_HS(model, zbins, T,
                        Vext_A, Vext_B, symmetric=False,
                        eps_AA=1, eps_BB=1, interaction_parameter=0,
                        sigma_AA=1, sigma_BB=1,
                        rc_AA = 2.5, rc_AB = 2.5, rc_BB = 2.5, shift = True,
                        rho_bulk_A = None, mu_A = None,
                        rho_bulk_B = None, mu_B = None,
                        initial_guess_A=0.04, initial_guess_B=0.04,
                        input_bins=1201,
                        plot=True, maxiter=100000, alpha_initial=0.000001, 
                        alpha_updates=None,
                        print_every=1000, plot_every=1000, tolerance=1e-5,
                        output_dict=False):
    """
    Calculate the density profile with neural DFT using a standard Picard iteration 
    for two types of particles for long-range interactions.

    Parameters:
    - model (tf.keras.Model): The Keras model to be used for the calculation of the 
                              one-body direct correlation function.
    - zbins (array-like): Spatial grid points.
    - Vext (array-like): Negative of the External potential in units of kbT (-beta Vext)
    - rho_bulk: bulk density (leave mu unspecified)
    - mu: chemical potential of long range system in units of kbT (beta mu)
    - interaction_parameter: diagonal element of array, where eps_ij = (1-interaction_parameter)*sqrt(eps_ii*eps_jj)
    - plot (bool): Toggle interactive plotting.
    - maxiter (int): Maximum number of Picard steps.
    - alpha_initial (float): Initial value for the relaxation parameter alpha.
    - alpha_updates (dict): Dictionary of iteration thresholds and corresponding 
                            alpha values to update alpha during iterations.
    - initial_guess (float): Initial guess for the density profile.

    Returns:
    - tuple: z coordinates and density profile.
    """
    """
    if (mu_A is None and rho_bulk_A is None) or (mu_A is not None and rho_bulk_A is not None):
        raise ValueError("Specify either the chemical potential 'mu' or the mean density 'rho_bulk' for species A")
    
    if (mu_B is None and rho_bulk_B is None) or (mu_B is not None and rho_bulk_B is not None):
        raise ValueError("Specify either the chemical potential 'mu' or the mean density 'rho_bulk' for species B")
    """
    T = T * np.ones_like(zbins)
    L = zbins[-1] - zbins[0]
    beta = 1/T # Assuming kb=1
    bin_width = abs(zbins[1] - zbins[0])

    eps_AB = (1-interaction_parameter) * np.sqrt(eps_AA*eps_BB)
    sigma_AB = 0.5 * (sigma_AA + sigma_BB)
    
    # setting up grid
    rho_A_new = np.zeros_like(zbins)
    rho_B_new = np.zeros_like(zbins)
    validA = np.isfinite(Vext_A) 
    validB = np.isfinite(Vext_B)
    rho_A = initial_guess_A * np.ones_like(zbins)
    rho_B = initial_guess_B * np.ones_like(zbins)
    log_rho_A_new = np.zeros_like(zbins)
    log_rho_B_new = np.zeros_like(zbins)
    log_rho_A = np.zeros_like(zbins)
    log_rho_B = np.zeros_like(zbins)
    log_rho_A[validA] = np.log(initial_guess_A)
    log_rho_B[validB] = np.log(initial_guess_B)
    log_rho_A[~validA] = -np.inf
    log_rho_B[~validB] = -np.inf
    
    potential_array_AA = lmft.attract(zbins, eps=eps_AA, sig=sigma_AA, cutoff=rc_AA, shift = shift)
    potential_array_AB = lmft.attract(zbins, eps=eps_AB, sig=sigma_AB, cutoff=rc_AB, shift = shift)
    potential_array_BB = lmft.attract(zbins, eps=eps_BB, sig=sigma_BB, cutoff=rc_BB, shift = shift)

    # Calculate delta mu
    """
    if rho_bulk_A is not None and rho_bulk_B is not None:
        rho_b_a = rho_bulk_A
        rho_b_b = rho_bulk_B
        mu_LR_A, mu_LR_B = eos.calc_mu_mixture(np.array([rho_b_a, rho_b_b]), T[0], np.array([eps_AA, eps_BB]),
                                               np.array([sigma_AA, sigma_BB]), np.array([[0, interaction_parameter], [interaction_parameter, 0]]))

    if mu_A is not None and mu_B is not None:
        mu_LR_A = mu_A*T[0]
        mu_LR_B = mu_B*T[0]
        rho_b_a, rho_b_b = eos.calc_rhob_mixture(np.array([mu_LR_A, mu_LR_B]), T[0], np.array([eps_AA, eps_BB]),
                                               np.array([sigma_AA, sigma_BB]), np.array([[0, interaction_parameter], [interaction_parameter, 0]]))
    """
    mu_LR_A = mu_A*T[0]
    mu_LR_B = mu_B*T[0]

    if rho_bulk_A is None and rho_bulk_B is None:
         # may give wrong densities when near phase coexistence
         rho_b_a, rho_b_b = eos.calc_rhob_mixture(np.array([mu_LR_A, mu_LR_B]), T[0], np.array([eps_AA, eps_BB]),
                                               np.array([sigma_AA, sigma_BB]), np.array([[0, interaction_parameter], [interaction_parameter, 0]]))
    else:
        rho_b_a = rho_bulk_A
        rho_b_b = rho_bulk_B

    rho_tot = rho_b_a + rho_b_b

    mu_R_A = np.log(rho_b_a) - np.mean(neural.c1_onetype(model, rho_tot*np.ones_like(T), input_bins)) # beta mu
    mu_correction_A = mu_LR_A - mu_R_A*T[0]
    mu_R_B = np.log(rho_b_b) - np.mean(neural.c1_onetype(model, rho_tot*np.ones_like(T), input_bins)) # beta mu
    mu_correction_B = mu_LR_B - mu_R_B*T[0]

    # Picard iteration parameter
    alpha = alpha_initial
    if alpha_updates is None:
        alpha_updates = alpha_updates_default_twotype
    
    if plot:
        fig, ax = plt.configure_plot(zbins)
        color_count = 0
  
    for i in range(maxiter + 1):
        if i in alpha_updates:
            alpha = alpha_updates[i]
        
        if plot and i % plot_every == 0:
            plt.plot_interactive_SR_twotype(fig, ax, zbins, rho_A, rho_B, Vext_A+beta*mu_LR_A, Vext_B+beta*mu_LR_B, color_count)
            color_count += 1
            
        # correlation from trained SR model
        c1_pred_SR = neural.c1_onetype(model, rho_A+rho_B, input_bins)
        
        V_correction_AA = lmft.V_correction(bin_width, rho_A - rho_b_a, potential_array_AA)
        V_correction_AB = lmft.V_correction(bin_width, rho_B - rho_b_b, potential_array_AB)
        V_correction_BA = lmft.V_correction(bin_width, rho_A - rho_b_a, potential_array_AB)
        V_correction_BB = lmft.V_correction(bin_width, rho_B - rho_b_b, potential_array_BB)

        
        c1_LR_A = c1_pred_SR + beta*(V_correction_AA + V_correction_AB) - beta*mu_correction_A
        c1_LR_B = c1_pred_SR + beta*(V_correction_BB + V_correction_BA) - beta*mu_correction_B

        # update density
        log_rho_A_new[validA] = Vext_A[validA] + beta[validA]*mu_LR_A + c1_LR_A[validA]
        log_rho_B_new[validB] = Vext_B[validB] + beta[validB]*mu_LR_B + c1_LR_B[validB]
        log_rho_A_new[~validA] = -np.inf
        log_rho_B_new[~validB] = -np.inf
        rho_A_new = np.exp(log_rho_A_new)
        rho_B_new = np.exp(log_rho_B_new)
        log_rho_A = (1 - alpha) * log_rho_A + alpha * log_rho_A_new
        log_rho_B = (1 - alpha) * log_rho_B + alpha * log_rho_B_new

        if symmetric == True:
            # Enforce reflection symmetry about center
            log_rho_A = 0.5 * (log_rho_A + log_rho_A[::-1])
            log_rho_B = 0.5 * (log_rho_B + log_rho_B[::-1])

        rho_A = np.exp(log_rho_A)
        rho_B = np.exp(log_rho_B)
    
        
        delta_A = np.max(np.abs(rho_A_new - rho_A))
        delta_B = np.max(np.abs(rho_B_new - rho_B))
        delta = max(delta_A, delta_B)
        
        if np.isnan(delta):
            print("Not converged: delta is NaN")
            return  None, None, None

        relative_error = delta / max(np.max(rho_B), np.max(rho_A))
        
        if i % print_every == 0:
            print(f"Iteration {i}: delta = {delta}")

        if delta < tolerance or relative_error < tolerance:
            print(f"Converged after {i} iterations (delta = {delta})")
            if plot:
                plt.plot_end_SR_twotype(zbins, rho_A, rho_B, Vext_A+beta*mu_LR_A, Vext_B+beta*mu_LR_B, ax)
            return zbins, rho_A, rho_B
        
    print(f"Not converged after {maxiter} iterations (delta = {delta})")

    return None, None, None #zbins, best_rho_A, best_rho_B


def minimise_LR_twotype_HS_mf(model, zbins, T,
                        Vext_A, Vext_B, symmetric=False,
                        eps_AA=1, eps_BB=1, interaction_parameter=0,
                        sigma_AA=1, sigma_BB=1,
                        rc_AA = 2.5, rc_AB = 2.5, rc_BB = 2.5, shift = True,
                        rho_bulk_A = None, mu_A = None,
                        rho_bulk_B = None, mu_B = None,
                        initial_guess_A=0.04, initial_guess_B=0.04,
                        input_bins=1201,
                        plot=True, maxiter=100000, alpha_initial=0.000001, 
                        alpha_updates=None,
                        print_every=1000, plot_every=1000, tolerance=1e-5,
                        output_dict=False):
    """
    Calculate the density profile with neural DFT using a standard Picard iteration 
    for two types of particles for long-range interactions.

    Parameters:
    - model (tf.keras.Model): The Keras model to be used for the calculation of the 
                              one-body direct correlation function.
    - zbins (array-like): Spatial grid points.
    - Vext (array-like): Negative of the External potential in units of kbT (-beta Vext)
    - rho_bulk: bulk density (leave mu unspecified)
    - mu: chemical potential of long range system in units of kbT (beta mu)
    - interaction_parameter: diagonal element of array, where eps_ij = (1-interaction_parameter)*sqrt(eps_ii*eps_jj)
    - plot (bool): Toggle interactive plotting.
    - maxiter (int): Maximum number of Picard steps.
    - alpha_initial (float): Initial value for the relaxation parameter alpha.
    - alpha_updates (dict): Dictionary of iteration thresholds and corresponding 
                            alpha values to update alpha during iterations.
    - initial_guess (float): Initial guess for the density profile.

    Returns:
    - tuple: z coordinates and density profile.
    """
    """
    if (mu_A is None and rho_bulk_A is None) or (mu_A is not None and rho_bulk_A is not None):
        raise ValueError("Specify either the chemical potential 'mu' or the mean density 'rho_bulk' for species A")
    
    if (mu_B is None and rho_bulk_B is None) or (mu_B is not None and rho_bulk_B is not None):
        raise ValueError("Specify either the chemical potential 'mu' or the mean density 'rho_bulk' for species B")
    """
    T = T * np.ones_like(zbins)
    L = zbins[-1] - zbins[0]
    beta = 1/T # Assuming kb=1
    bin_width = abs(zbins[1] - zbins[0])

    eps_AB = (1-interaction_parameter) * np.sqrt(eps_AA*eps_BB)
    sigma_AB = 0.5 * (sigma_AA + sigma_BB)
    
    # setting up grid
    rho_A_new = np.zeros_like(zbins)
    rho_B_new = np.zeros_like(zbins)
    validA = np.isfinite(Vext_A) 
    validB = np.isfinite(Vext_B)
    rho_A = initial_guess_A * np.ones_like(zbins)
    rho_B = initial_guess_B * np.ones_like(zbins)
    log_rho_A_new = np.zeros_like(zbins)
    log_rho_B_new = np.zeros_like(zbins)
    log_rho_A = np.zeros_like(zbins)
    log_rho_B = np.zeros_like(zbins)
    log_rho_A[validA] = np.log(initial_guess_A)
    log_rho_B[validB] = np.log(initial_guess_B)
    log_rho_A[~validA] = -np.inf
    log_rho_B[~validB] = -np.inf
    
    potential_array_AA = lmft.attract(zbins, eps=eps_AA, sig=sigma_AA, cutoff=rc_AA, shift = shift)
    potential_array_AB = lmft.attract(zbins, eps=eps_AB, sig=sigma_AB, cutoff=rc_AB, shift = shift)
    potential_array_BB = lmft.attract(zbins, eps=eps_BB, sig=sigma_BB, cutoff=rc_BB, shift = shift)

    mu_LR_A = mu_A*T[0]
    mu_LR_B = mu_B*T[0]

    if rho_bulk_A is None and rho_bulk_B is None:
         # may give wrong densities when near phase coexistence
         rho_b_a, rho_b_b = eos.calc_rhob_mixture(np.array([mu_LR_A, mu_LR_B]), T[0], np.array([eps_AA, eps_BB]),
                                               np.array([sigma_AA, sigma_BB]), np.array([[0, interaction_parameter], [interaction_parameter, 0]]))
    else:
        rho_b_a = rho_bulk_A
        rho_b_b = rho_bulk_B

    # Picard iteration parameter
    alpha = alpha_initial
    if alpha_updates is None:
        alpha_updates = alpha_updates_default_twotype
    
    if plot:
        fig, ax = plt.configure_plot(zbins)
        color_count = 0
  
    for i in range(maxiter + 1):
        if i in alpha_updates:
            alpha = alpha_updates[i]
        
        if plot and i % plot_every == 0:
            plt.plot_interactive_SR_twotype(fig, ax, zbins, rho_A, rho_B, Vext_A+beta*mu_LR_A, Vext_B+beta*mu_LR_B, color_count)
            color_count += 1
            
        # correlation from trained SR model
        c1_pred_SR = neural.c1_onetype(model, rho_A+rho_B, input_bins)
        
        V_correction_AA = lmft.V_correction(bin_width, rho_A, potential_array_AA)
        V_correction_AB = lmft.V_correction(bin_width, rho_B, potential_array_AB)
        V_correction_BA = lmft.V_correction(bin_width, rho_A, potential_array_AB)
        V_correction_BB = lmft.V_correction(bin_width, rho_B, potential_array_BB)

        
        c1_LR_A = c1_pred_SR + beta*(V_correction_AA + V_correction_AB)
        c1_LR_B = c1_pred_SR + beta*(V_correction_BB + V_correction_BA)

        # update density
        log_rho_A_new[validA] = Vext_A[validA] + beta[validA]*mu_LR_A + c1_LR_A[validA]
        log_rho_B_new[validB] = Vext_B[validB] + beta[validB]*mu_LR_B + c1_LR_B[validB]
        log_rho_A_new[~validA] = -np.inf
        log_rho_B_new[~validB] = -np.inf
        rho_A_new = np.exp(log_rho_A_new)
        rho_B_new = np.exp(log_rho_B_new)
        log_rho_A = (1 - alpha) * log_rho_A + alpha * log_rho_A_new
        log_rho_B = (1 - alpha) * log_rho_B + alpha * log_rho_B_new

        if symmetric == True:
            # Enforce reflection symmetry about center
            log_rho_A = 0.5 * (log_rho_A + log_rho_A[::-1])
            log_rho_B = 0.5 * (log_rho_B + log_rho_B[::-1])

        rho_A = np.exp(log_rho_A)
        rho_B = np.exp(log_rho_B)
    
        
        delta_A = np.max(np.abs(rho_A_new - rho_A))
        delta_B = np.max(np.abs(rho_B_new - rho_B))
        delta = max(delta_A, delta_B)
        
        if np.isnan(delta):
            print("Not converged: delta is NaN")
            return  None, None, None

        relative_error = delta / max(np.max(rho_B), np.max(rho_A))
        
        if i % print_every == 0:
            print(f"Iteration {i}: delta = {delta}")

        if delta < tolerance or relative_error < tolerance:
            print(f"Converged after {i} iterations (delta = {delta})")
            if plot:
                plt.plot_end_SR_twotype(zbins, rho_A, rho_B, Vext_A+beta*mu_LR_A, Vext_B+beta*mu_LR_B, ax)
            return zbins, rho_A, rho_B
        
    print(f"Not converged after {maxiter} iterations (delta = {delta})")

    return None, None, None #zbins, best_rho_A, best_rho_B