import numpy as np
import tensorflow as tf
from tensorflow import keras
import csv


# Enable or disable Tensor Float 32 Execution
tf.config.experimental.enable_tensor_float_32_execution(False)


def generate_windows(array, bins):
    """
    Generate sliding windows for the input array with a given bin size.

    Parameters:
    - array (np.ndarray): Input array.
    - bins (int): Number of bins on each side of the central bin.
    - mode (str): Padding mode for np.pad (default is "wrap").

    Returns:
    - np.ndarray: Array of sliding windows.
    """
    padded_array = np.pad(array, bins, mode="wrap")
    windows = np.empty((len(array), 2 * bins + 1))
    for i in range(len(array)):
        windows[i] = padded_array[i:i + 2 * bins + 1]
    return windows

def c1_onetype(model, density_profile, input_bins, dx=0.005, return_c2=False, output_dict=False):
    """
    Infer the one-body direct correlation profile from a given density profile 
    using a neural correlation functional.

    Parameters:
    - model (tf.keras.Model): The neural correlation functional.
    - density_profile (np.ndarray): The density profile.
    - dx (float): The discretization of the input layer of the model.
    - input_bins (int): Number of input bins for the model.
    - return_c2 (bool or str): If False, only return c1(x). If True, return both 
                               c1 as well as the corresponding two-body direct 
                               correlation function c2(x, x') which is obtained 
                               via autodifferentiation. If 'unstacked', give c2 
                               as a function of x and x-x', i.e., as obtained 
                               naturally from the model.

    Returns:
    - np.ndarray: c1(x) or (c1(x), c2(x, x')) depending on the value of return_c2.
    """
    window_bins = (input_bins - 1) // 2
    rho_windows = generate_windows(density_profile, window_bins).reshape(density_profile.shape[0], input_bins, 1)
    
    if return_c2:
        rho_windows = tf.Variable(rho_windows)
        with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
            tape.watch(rho_windows)
            result = model(rho_windows)
        jacobi_windows = tape.batch_jacobian(result, rho_windows).numpy().squeeze() / dx
        c1_result = result.numpy().flatten()
        
        if return_c2 == "unstacked":
            return c1_result, jacobi_windows
        
        c2_result = np.row_stack([
            np.roll(np.pad(jacobi_windows[i], (0, density_profile.shape[0] - input_bins)), i - window_bins) 
            for i in range(density_profile.shape[0])
        ])
        return c1_result, c2_result
    
    if output_dict:
        return model.predict_on_batch(rho_windows)["c1"].flatten()
    
    return model.predict_on_batch(rho_windows).flatten()

def c1_onetype_T(model, density_profile, T, input_bins, dx=0.005, return_c2=False, output_dict=False):
    """
    Infer the one-body direct correlation profile from a given density profile 
    using a neural correlation functional.

    Parameters:
    - model (tf.keras.Model): The neural correlation functional.
    - density_profile (np.ndarray): The density profile.
    - T (float): Temperature.
    - dx (float): The discretization of the input layer of the model.
    - input_bins (int): Number of input bins for the model.
    - return_c2 (bool or str): If False, only return c1(x). If True, return both 
                               c1 as well as the corresponding two-body direct 
                               correlation function c2(x, x') which is obtained 
                               via autodifferentiation. If 'unstacked', give c2 
                               as a function of x and x-x', i.e., as obtained 
                               naturally from the model.

    Returns:
    - np.ndarray: c1(x) or (c1(x), c2(x, x')) depending on the value of return_c2.
    """
    # Same as c1_pred from training script
    window_bins = (input_bins - 1) // 2
    rho_windows = generate_windows(density_profile, window_bins).reshape(density_profile.shape[0], input_bins, 1)
    
    if return_c2:
        rho_windows = tf.Variable(rho_windows)
        with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
            tape.watch(rho_windows)
            result = model(rho_windows)
        jacobi_windows = tape.batch_jacobian(result, rho_windows).numpy().squeeze() / dx
        c1_result = result.numpy().flatten()
        
        if return_c2 == "unstacked":
            return c1_result, jacobi_windows
        
        c2_result = np.row_stack([
            np.roll(np.pad(jacobi_windows[i], (0, density_profile.shape[0] - input_bins)), i - window_bins) 
            for i in range(density_profile.shape[0])
        ])
        return c1_result, c2_result
    
    if output_dict:
        return model.predict_on_batch([rho_windows, T])["c1"].flatten()
    
    return model.predict_on_batch([rho_windows, T]).flatten()



def c1_onetype_params(model, density_profile, params, input_bins, dx=0.005, return_c2=False, output_dict=False):
    """
    Infer the one-body direct correlation profile from a given density profile 
    using a neural correlation functional.

    Parameters:
    - model (tf.keras.Model): The neural correlation functional.
    - density_profile (np.ndarray): The density profile.
    - params: A Dict with additional parameters that are required as input to the model (e.g. temperature).
    - dx (float): The discretization of the input layer of the model.
    - input_bins (int): Number of input bins for the model.
    - return_c2 (bool or str): If False, only return c1(x). If True, return both 
                               c1 as well as the corresponding two-body direct 
                               correlation function c2(x, x') which is obtained 
                               via autodifferentiation. If 'unstacked', give c2 
                               as a function of x and x-x', i.e., as obtained 
                               naturally from the model.

    Returns:
    - np.ndarray: c1(x) or (c1(x), c2(x, x')) depending on the value of return_c2.
    """
    # Same as c1_pred from training script
    window_bins = (input_bins - 1) // 2
    rho_windows = generate_windows(density_profile, window_bins).reshape(density_profile.shape[0], input_bins, 1)
    paramsInput = {key: tf.convert_to_tensor(np.full(density_profile.shape[0], value)) for key, value in params.items()}

    if return_c2:
        rho_windows = tf.Variable(rho_windows)
        with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
            tape.watch(rho_windows)
            result = model(rho_windows, **paramsInput)
        jacobi_windows = tape.batch_jacobian(result, rho_windows).numpy().squeeze() / dx
        c1_result = result.numpy().flatten()
        
        if return_c2 == "unstacked":
            return c1_result, jacobi_windows
        
        c2_result = np.row_stack([
            np.roll(np.pad(jacobi_windows[i], (0, density_profile.shape[0] - input_bins)), i - window_bins) 
            for i in range(density_profile.shape[0])
        ])
        return c1_result, c2_result
    
    if output_dict:
        return model.predict_on_batch([rho_windows, *paramsInput.values()])["c1"].flatten()
    
    return model.predict_on_batch([rho_windows, *paramsInput.values()]).flatten()



def c1_pred_twotype(model, rho_A, rho_B, sigma_AA, sigma_BB, input_bins=1201):

    window_bins = (input_bins - 1) // 2
    rho_A_windows = generate_windows(rho_A, window_bins).reshape(rho_A.shape[0], input_bins, 1)
    rho_B_windows = generate_windows(rho_B, window_bins).reshape(rho_B.shape[0], input_bins, 1)

    c1_result_A = model.predict_on_batch([rho_A_windows, rho_B_windows, sigma_AA, sigma_BB])[0].flatten()
    c1_result_B = model.predict_on_batch([rho_A_windows, rho_B_windows, sigma_AA, sigma_BB])[1].flatten()
    
    return c1_result_A, c1_result_B


def write_profile(filename, centers, densities):
    """
    Write the density profile to a file.

    Parameters:
    - filename (str): Output file name.
    - centers (np.ndarray): Bin centers.
    - densities (np.ndarray): Density values.
    """
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=' ')
        writer.writerow(["xbins", "rho"])
        for center, density in zip(centers, densities):
            writer.writerow([f"{center:.4f}", f"{density:.20f}"])
