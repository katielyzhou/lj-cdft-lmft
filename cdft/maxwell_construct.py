import numpy as np
from scipy.signal import argrelextrema
from scipy.integrate import simpson
from scipy.interpolate import interp1d

def spinodal_idx(chemical_potential):
    # Return the indices corresponding to the max and min mu in the vdw loop

    # Find the local minima and maxima (for the Van der Waals loop)
    minima = argrelextrema(chemical_potential, np.less)[0]
    maxima = argrelextrema(chemical_potential, np.greater_equal)[0]

    # Extract the first minimum and maximum to form the Van der Waals loop
    loop_min_idx = minima[0]
    loop_max_idx = maxima[0]

    return loop_max_idx, loop_min_idx

def roots(density, chemical_potential, target_mu, tol=1e-5, max_iter=100):
    # Newton-Raphson method to calculate where the coexistence mu crosses the equation of state
    # to find rho_liq and rho_gas

    # Interpolation of mu vs density
    mu_interp = interp1d(density, chemical_potential, kind="cubic", fill_value="extrapolate")
    
    # Derivative of the interpolated mu with respect to density
    mu_deriv_interp = interp1d(density, np.gradient(chemical_potential, density), kind="cubic", fill_value="extrapolate")
    
    # Function for finding roots
    def f(rho):
        return mu_interp(rho) - target_mu
    
    # Derivative of the function
    def df(rho):
        return mu_deriv_interp(rho)
    
    # Initial guesses: Take initial points near where the sign change occurs
    intersection_indices = np.where(np.diff(np.sign(mu_interp(density) - target_mu)))[0]
    if len(intersection_indices) < 3:
        return np.array([]), np.array([])  # No roots found
    
    rho_intersections = []
    mu_intersections = []
    
    # Try multiple starting points based on where sign change occurs
    for idx in intersection_indices:
        initial_guess = density[idx]
        rho = initial_guess
        
        for _ in range(max_iter):
            # Newton-Raphson update
            f_value = f(rho)
            df_value = df(rho)
            
            if np.abs(df_value) < 1e-10:  # Avoid division by zero
                break
            
            # Update the guess
            rho_new = rho - f_value / df_value
            
            if np.abs(rho_new - rho) < tol:  # Convergence check
                rho_intersections.append(rho_new)
                mu_intersections.append(mu_interp(rho_new))
                break
            
            # Update rho for the next iteration
            rho = rho_new
    
    return np.array(rho_intersections)


def calc_area(density, chemical_potential, density_roots, target_mu):
    density_roots.sort()

    # Interpolate the chemical potential as a continuous function of density
    interpolation = interp1d(density, chemical_potential, kind='cubic', fill_value='extrapolate')
    
    # Define the density ranges based on roots
    first_root, second_root, third_root = density_roots
    
    # Generate a finer density grid to perform integration
    fine_density1 = np.linspace(first_root, second_root, 50)
    fine_density2 = np.linspace(second_root, third_root, 50)
    
    # Interpolate chemical potential over the finer density grid
    interpolated_mu1 = interpolation(fine_density1)
    interpolated_mu2 = interpolation(fine_density2)
    
    # Calculate the area difference using Simpson's rule
    A_1 = simpson(interpolated_mu1 - target_mu, x=fine_density1)
    A_2 = simpson(target_mu - interpolated_mu2, x=fine_density2)

    deltaA = A_1 - A_2
    
    return deltaA, fine_density1, fine_density2, interpolated_mu1, interpolated_mu2


def calc_mu_coex(density, chemical_potential, tol=1e-4, iter=100000):
    # tol is tolerance in area
    
    idx1, idx2 = spinodal_idx(chemical_potential)
    
    mu_guess = 0.5*(chemical_potential[idx1] + chemical_potential[idx2])
    # mean of local min and max

    for i in range(iter):
        
        delta = calc_area(density, chemical_potential, roots(density, chemical_potential, mu_guess), mu_guess)[0]
        
        if np.abs(delta) < tol:
            break

        elif i == iter-1:
            print("Maxwell mu has not converged; delta = {}".format(delta))

        elif delta > 0:
            mu_guess += 1e-5

        elif delta < 0:
            mu_guess -= 1e-5

    #print(f"mu_coex = {mu_guess}; area difference = {delta}")
    
    return mu_guess


def extract_rho_coex(density, chemical_potential):
    mu_coex = calc_mu_coex(density, chemical_potential)
    #print(mu_coex)
    root = roots(density, chemical_potential, mu_coex)
    root.sort()

    areas = calc_area(density, chemical_potential, root, mu_coex)[1:]

    return root, mu_coex, areas

