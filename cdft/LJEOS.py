import numpy as np
from feos.eos import EquationOfState, State, Contributions
from feos.pets import PetsParameters
import si_units as si
from scipy.interpolate import interp1d
import maxwell_construct as maxwell
from scipy.optimize import minimize

# Johnson's Equation of State

x = np.array([0.8623085097507421, 2.976218765822098, -8.402230115796038, 0.1054136629203555, -0.8564583828174598,
              1.582759470107601, 0.7639421948305453, 1.753173414312048, 2.798291772190376e3, -4.8394220260857657e-2,
              0.9963265197721935, -3.698000291272493e1, 2.084012299434647e1, 8.305402124717285e1, -9.574799715203068e2,
              -1.477746229234994e2, 6.398607852471505e1, 1.603993673294834e1, 6.805916615864377e1, -2.791293578795945e3,
              -6.245128304568454, -8.116836104958410e3, 1.488735559561229e1, -1.059346754655084e4, -1.131607632802822e2,
              -8.867771540418822e3, -3.986982844450543e1, -4.689270299917261e3, 2.593535277438717e2, -2.694523589434903e3,
              -7.218487631550215e2, 1.721802063863269e2])

################################

def ais(t):
    a = np.zeros(8)

    a[0] = x[1-1]*t + x[2-1]*np.sqrt(t) + x[3-1] + x[4-1]/t + x[5-1]/t**2
    a[1] = x[6-1]*t + x[7-1] + x[8-1]/t + x[9-1]/t**2
    a[2] = x[10-1]*t + x[11-1] + x[12-1]/t
    a[3] = x[13-1]
    a[4] = x[14-1]/t + x[15-1]/t**2
    a[5] = x[16-1]/t
    a[6] = x[17-1]/t + x[18-1]/t**2
    a[7] = x[19-1]/t**2

    return a

################################

def bis(t):
    b = np.zeros(6)

    b[0] = x[20-1]/t**2 + x[21-1]/t**3
    b[1] = x[22-1]/t**2 + x[23-1]/t**4
    b[2] = x[24-1]/t**2 + x[25-1]/t**3
    b[3] = x[26-1]/t**2 + x[27-1]/t**4
    b[4] = x[28-1]/t**2 + x[29-1]/t**3
    b[5] = x[30-1]/t**2 + x[31-1]/t**3  + x[32-1]/t**4
    
    return b

################################

def Gis(rho):
    gamma=3.0
    F = np.exp(-gamma*rho**2)

    G = np.zeros(6)

    G[0] = (1-F)/(2*gamma)
    G[1] = -(F*rho**2 - 2*G[1-1])/(2*gamma)
    G[2] = -(F*rho**4 - 4*G[2-1])/(2*gamma)
    G[3] = -(F*rho**6 - 6*G[3-1])/(2*gamma)
    G[4] = -(F*rho**8 - 8*G[4-1])/(2*gamma)
    G[5] = -(F*rho**10 - 10*G[5-1])/(2*gamma)            
    
    return G

################################

def calcAr(rho,t):
    # Excess Helmholtz free energy
    Ar = 0.0
    a = ais(t)
    b = bis(t)
    G = Gis(rho)
    
    for ii in range(8):
        i = ii+1
        Ar += a[ii]*(rho**i)/i
    
    for ii in range(6):
        i = ii+1
        Ar += b[ii]*G[ii]

    return Ar

def calcP(rho,t):
    # Pressure in reduced units
    P = rho*t
    a = ais(t)
    b = bis(t)
    gamma=3.0
    F = np.exp(-gamma*rho**2)

    for ii in range(8):
        i = ii+1
        P += a[ii]*(rho**(i+1))

    for ii in range(6):
        i = ii+1
        P += F*b[ii]*rho**(2*i+1)

    return P


def MeanFieldDiff(rho,sig,rc):
    return -(32.0/9.0)*np.pi*rho*( (sig/rc)**9 - 1.5*(sig/rc)**3)

def calc_mu(rho, t, eps=1, sig=1, rc=2.5):
    # Given a bulk density rho and temperature t, returns mu

    if rc == 2.5: # Use PeTS
    
        epsilon_k = eps * si.KELVIN
        sigma = sig * si.ANGSTROM

        parameters = PetsParameters.from_values(sigma/si.ANGSTROM, epsilon_k/si.KELVIN)
        pets = EquationOfState.pets(parameters)

        s = State(
        eos=pets,
        temperature=t*epsilon_k,
        volume=1000*sigma**3,
        density= rho /si.NAV / sigma**3,
        )

        mu = s.chemical_potential(Contributions.Residual) / (si.RGAS* t*epsilon_k) + np.log(rho) # beta mu

        return mu*t #assuming kb=1
    
    elif rc == None: # Use Johnson's equation of state for full LJ
        P = calcP(rho, t)
        mu_r = calcAr(rho, t) + P/rho - t
        mu_id = t*np.log(rho)
        return mu_id + mu_r
    
    elif rc == "J2.5": # Use Johnson's equation of state for rc=2.5
        rc = 2.5
        P = calcP(rho, t)
        mu_r = calcAr(rho, t) + 2*MeanFieldDiff(rho, sig, rc) + P/rho - t
        # One meanfield diff for Ar, other for P
        mu_id = t*np.log(rho)
        return mu_id + mu_r
    
    else: # Use Johnson's equation of state otherwise

        P = calcP(rho, t)
        mu_r = calcAr(rho, t) + 2*MeanFieldDiff(rho, sig, rc) + P/rho - t
        # One meanfield diff for Ar, other for P
        mu_id = t*np.log(rho)
        return mu_id + mu_r
        

def calc_rhob(betamu, t, eps=1, sig=1, rc=2.5):
    """
    betamu (float): chemical potential in units of kbT
    t (float): temperature

    """
    rhob_range = np.concatenate((np.logspace(-3, -1, num=30, base=10), np.linspace(0.11, 1, 50)))
    mu_range = np.array([calc_mu(i, t, eps, sig, rc) for i in rhob_range]).reshape(rhob_range.shape) # mu not beta mu
    mu_range_new = np.empty_like(mu_range)

    try:
        # Perform a Maxwell construction to obtain the coexistence densities
        roots, mu_coex, areas = maxwell.extract_rho_coex(rhob_range, mu_range)
    except IndexError: # When Maxwell construction fails (e.g. above Tc)
        rho_interp = interp1d(mu_range/t, rhob_range, kind='linear', bounds_error=False, fill_value=np.nan)
        return rho_interp(betamu)
    
    # When Maxwell construction works
    rho_g = roots[0]
    rho_l = roots[-1]

    # Make a new equation of state that removes the vdw loop
    mu_range_new[rhob_range < rho_g] = mu_range[rhob_range < rho_g]
    mu_range_new[rhob_range > rho_l] = mu_range[rhob_range > rho_l]
    unphysical = (rhob_range >= rho_g) & (rhob_range <= rho_l)
    mu_range_new[unphysical] = mu_coex

    mu_range_new = mu_range_new/t # To get in units of kBT

    rho_interp = interp1d(mu_range_new, rhob_range, kind='linear', bounds_error=False, fill_value=np.nan)

    return rho_interp(betamu)

################################

def calc_mu_mixture(bulk_densities, t, eps_array, sig_array, k_ij):
    # k_ij is the matrix of binary interaction parameters, such that eps_ij = (1 - k_ij) * sqrt(eps_ii*eps_jj)
    # arrays must be of floats not integers

    epsilon_k = eps_array * si.KELVIN
    sigma = sig_array * si.ANGSTROM
    vol = 1000 # doesn't really matter, just needs to be consistent

    parameters = PetsParameters.from_lists(sigma = sigma/si.ANGSTROM, epsilon_k = epsilon_k/si.KELVIN, k_ij = k_ij)
    pets = EquationOfState.pets(parameters)

    s = State(
        eos=pets,
        temperature=t*epsilon_k[0],
        volume=vol*si.ANGSTROM**3,
        #moles = bulk_densities / si.NAV * vol,
        total_moles = np.sum(bulk_densities) * vol / si.NAV,
        molefracs = bulk_densities/np.sum(bulk_densities)
        )

    betamu = s.chemical_potential(Contributions.Residual) / (si.RGAS* t*epsilon_k[0]) + np.log(bulk_densities)
    # pets chemical potential result is an array of excess chemical potential of each component


    return betamu*t  #assuming kb=1


def calc_rhob_mixture(target_mu, t, eps_array, sig_array, k_ij, guess=None):
    ### If near phase coexistence can give the wrong bulk density
    """
    Given target chemical potentials (target_mu), invert the EOS to get bulk_densities.
    Inputs:
        - target_mu: array of target chemical potentials (not betamu)
        - t: reduced temperature (in units of eps_ii)
        - eps_array, sig_array: Lennard-Jones parameters
        - k_ij: binary interaction matrix
        - guess: array of optional initial guess for densities
    Returns:
        - bulk_densities: array of densities corresponding to target chemical potentials
    """

    if guess is None:
        guess = np.array([1e-3, 1e-3])  # default low-density guess

    def objective(dens):
        # Avoid non-physical densities
        if np.any(dens <= 0):
            return 1e10

        try:
            mu_calc = calc_mu_mixture(dens, t, eps_array, sig_array, k_ij)
            return np.sum((mu_calc - target_mu)**2)
        except:
            return 1e10  # penalize if evaluation fails

    result = minimize(objective, guess, method='Nelder-Mead', options={
                                                                'xatol': 1e-6,
                                                                'fatol': 1e-6,
                                                                'maxiter': 1000})

    if not result.success:
        raise RuntimeError("Optimization failed: " + result.message)

    return result.x
