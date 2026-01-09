import numpy as np

eps = 1
sig = 1
rc = 2.5*sig

def attract(xbin, eps=1, sig=1, cutoff=2.5, shift = True):
    # Returns the planar attractive interactions to calculate V correction.

    r_min = 2**(1/6)*sig
    L = xbin[-1] - xbin[0]
    xbin_extend = np.concatenate((xbin - L, xbin, xbin+L))
    z = np.abs(xbin_extend)
    interactions = np.zeros((len(z)))

    if cutoff == None: # Full LJ

        f1 = np.pi*eps*((4/5)*sig**12*(r_min**(-10)) - 2*sig**6*(r_min**(-4))+ z**2 - r_min**2)

        f2 = np.pi*eps*((4/5)*sig**12*(z**(-10)) - 2*sig**6*(z**(-4)))

        np.putmask(interactions, (z <= r_min), f1)
        np.putmask(interactions, (r_min < z), f2)

    else:
        if cutoff=="J2.5":
            cutoff = 2.5

        V_LJ = 4*eps*((sig/cutoff)**12 - (sig/cutoff)**6)

        if shift == True: # Truncated and shifted

            f1 = np.pi*eps*((4/5)*sig**12*(r_min**(-10)-cutoff**(-10)) - 2*sig**6*(r_min**(-4)-cutoff**(-4))+ z**2 - r_min**2) + np.pi*V_LJ*(z**2 - cutoff**2)

            f2 = np.pi*eps*((4/5)*sig**12*(z**(-10) - cutoff**(-10)) - 2*sig**6*(z**(-4) - cutoff**(-4))) + np.pi*V_LJ*(z**2 - cutoff**2)

        elif shift == False: # Truncated

            f1 = np.pi*eps*((4/5)*sig**12*(r_min**(-10)-cutoff**(-10)) - 2*sig**6*(r_min**(-4)-cutoff**(-4))+ z**2 - r_min**2)

            f2 = np.pi*eps*((4/5)*sig**12*(z**(-10) - cutoff**(-10)) - 2*sig**6*(z**(-4) - cutoff**(-4)))

        np.putmask(interactions, (z <= r_min), f1)
        np.putmask(interactions, (r_min < z) & (z <= cutoff), f2)
        

    return interactions

 
def V_correction(bin_width, rho, u1):

    padded_rho = np.concatenate((rho, rho, rho)) # Pad rho on both sides (like periodic boundary conditions)

    if len(u1) != len(padded_rho): # Can sometimes get arrays 1 off from each other
        diff = np.abs(len(u1) - len(padded_rho))
        
        # If u1 is longer than padded_rho, pad rho with its edges
        if len(u1) > len(padded_rho):
            pad_left = diff // 2
            pad_right = diff - pad_left
            padded_rho = np.concatenate((rho[-pad_left:], padded_rho, rho[:pad_right]))
        
        # If rho is longer than u1, pad u1 with zeros
        elif len(u1) < len(padded_rho):
            pad_left = diff // 2
            pad_right = diff - pad_left
            u1 = np.concatenate((np.zeros(pad_left), u1, np.zeros(pad_right)))

    frho = np.fft.fft(padded_rho)
    fatt = np.fft.fft(u1)
    conv = np.fft.ifft(frho*fatt)
    deltaV = -(conv.real*bin_width)[len(rho):-len(rho)] # negative sign to match definition (deltaphi - deltaphi_R)

    return deltaV
