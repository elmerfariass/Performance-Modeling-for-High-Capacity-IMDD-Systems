import numpy as np
from scipy.constants import k as kB  # Boltzmann constant

# Physical constant
q = 1.602176634e-19 # Electron charge [C]

def calc_S_th(N_0, fq_array):
    """
    Calculates the PSD of thermal noise, assumed to be constant across frequency.
    Based on S_th = N_0 / 2.
    """
    return np.full_like(fq_array, N_0 / 2)

def calc_S_th_opticompy(Tc, RL, fq_array):
    """
    Calculates the PSD of thermal noise.
    Uses OpticommPy parameters to internally calculate N_0,
    then applies the analytical equation S_th = N_0 / 2.
    
    Parameters:
    - Tc: Temperature in Celsius
    - RL: Load Resistance in Ohms
    - fq_array: Frequency array for the PSD (double-sided)
    """
    
    #Convert temperature from Celsius to Kelvin
    T = Tc + 273.15
    
    # Calculate N_0 based on the OpticommPy noise variance definition
    # N_0 = 4 * kB * T / RL
    N_0 = (4 * kB * T) / RL
    
    # Calculate the double-sided PSD (S_th = N_0 / 2)
    S_th = N_0 / 2
    
    #Return the constant PSD array across all frequencies
    return np.full_like(fq_array, S_th)


def calc_S_shot(P_RX, G, F, R, fq_array):
    """
    Calculates the PSD of shot noise.
    Assuming k_shot = (G^2 * F * q )/ R.
    """
    k_shot = (G**2 * F * q) / R
    S_shot = k_shot * P_RX
    return np.full_like(fq_array, S_shot)

def calc_S_shot_opticompy(P_RX, R, Id, fq_array):
    """
    Calculates the PSD of shot noise.
    Adapted to exactly match OpticommPy's time-domain generation.
    OpticommPy calculates variance as: Fs * q * (ipd + Id)
    Therefore, the double-sided PSD is q * (I_avg + Id).
    
    Note on APD vs PIN Photodiodes:
    The reference article defines the APD shot noise proportionality factor
    as k_shot = (G^2 * F * q) / R. However, the article also explicitly states 
    that for a standard PIN photodiode, G = 1 and F = 1. Since OpticommPy's 
    time-domain implementation does not apply avalanche gain (modeling a PIN), 
    we safely omit G and F to perfectly mirror the simulator's behavior.
    
    Parameters:
    - P_RX: Average received optical power (W)
    - R: Photodiode responsivity (A/W)
    - Id: Dark current (A)
    - fq_array: Frequency array for the PSD (double-sided)
    """
    
    # Calculate the average photocurrent based on received power
    I_avg = P_RX * R
    
    # Calculate the double-sided PSD matching OpticommPy's exact formula
    # S_shot = q * (photocurrent + dark_current)
    # (G and F are inherently 1 here, representing a PIN detector)
    S_shot = q * (I_avg + Id)
    
    #Return the constant PSD array across all frequencies
    return np.full_like(fq_array, S_shot)

def calc_S_RIN(P_TX_sq_avg, RIN_coeff_dB, fq_array ):
    """
    Calculates the PSD of RIN at the source.
    k_RIN = RIN_coeff_linear / 2.
    Returns a scalar, as the channel filter will be applied in the main function.
    """
    # Convert from dB/Hz to linear scale
    RIN_coeff_lin = 10**(RIN_coeff_dB / 10)
    k_RIN = RIN_coeff_lin / 2
    return np.full_like(fq_array, k_RIN * P_TX_sq_avg)

def calc_S_RIN_opticompy(RIN_var, Fs, fq_array):
    """
    Calculates the PSD of RIN.
    Adapted to exactly match OpticommPy's time-domain generation.
    OpticommPy simply generates white noise with a fixed variance (RIN_var).
    Therefore, the double-sided PSD is the variance divided by Fs.
    
    Parameters:
    - RIN_var: The variance of the RIN noise fed into OpticommPy
    - Fs: Sampling frequency (Hz)
    - fq_array: Frequency array for the PSD (double-sided)
    """
    
    # Calculate the double-sided PSD matching OpticommPy's generation
    S_RIN = RIN_var / Fs
    
    # Return the constant PSD array across all frequencies
    return np.full_like(fq_array, S_RIN)

def calc_S_ADC(PAPR, sigma_x_sq, ENOB, fs, fq_array):
    """
    Calculates the PSD of ADC quantization noise.
    Based on the quantization noise model.
    """
    S_ADC = ((PAPR) * sigma_x_sq / (12 * fs) * (2**(2 * ENOB - 2)))
    return np.full_like(fq_array, S_ADC)

def calc_S_ADC_opticompy(Vmax, Vmin, ENOB, outFs, fq_array):
    """
    Calculates the PSD of ADC quantization noise.
    Uses OpticommPy parameters to derive the exact analytical 
    equation: S_ADC = (PAPR * sigma_x^2) / (12 * fs * 2^(2*ENOB - 2))
    
    Parameters:
    - Vmax: Maximum voltage limit of the ADC clipping
    - Vmin: Minimum voltage limit of the ADC clipping
    - ENOB: Effective Number of Bits
    - outFs: Output sampling frequency of the ADC (Hz) - corresponds to fs
    - fq_array: Frequency array for the PSD (double-sided)
    """
    
    #  Find the peak voltage from the OpticommPy full-scale range
    Vpeak = (Vmax - Vmin) / 2
    
    #  The equation's numerator (PAPR * sigma_x^2) is mathematically equal to Vpeak^2
    PAPR_sigma_x_sq = Vpeak**2
    
    # Apply the exact equation as presented in the article/image
    # S_ADC(f) = (PAPR * sigma_x^2) / (12 * fs * 2^(2*ENOB - 2))
    S_ADC = PAPR_sigma_x_sq / (12 * outFs * (2**(2 * ENOB - 2)))
    
    return np.full_like(fq_array, S_ADC)

def calc_S_N(S_RIN_scalar, S_shot_array, S_th_array, S_ADC_array, H_ch_sq_array):
    """
    Consolidate the total noise S_N(f) at the equalizer input (Eq. 8).
    S_N(f) = S_RIN * |H_ch(f)|^2 + S_shot(f) + S_th(f) + S_ADC(f)
    """

    S_N_array = (S_RIN_scalar * H_ch_sq_array) + S_shot_array + S_th_array + S_ADC_array

    return S_N_array

