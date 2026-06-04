import numpy as np

# Physical constant
q = 1.602176634e-19 # Electron charge [C]

def calc_S_th(N_0, f_array):
    """
    Calculate the thermal noise PSD, assumed constant across frequency.
    Based on S_th = N_0 / 2.
    """
    return np.full_like(f_array, N_0 / 2)

def calc_S_shot(P_RX, G, F, R, f_array):
    """
    Calculate the shot noise PSD.
    Assuming k_shot = G^2 * F * q / R.
    """
    k_shot = (G**2 * F * q) / R
    S_shot_val = k_shot * P_RX
    return np.full_like(f_array, S_shot_val)

def calc_S_RIN(P_TX_sq_avg, RIN_coeff_dB):
    """
    Calculate the RIN PSD at the source.
    k_RIN = RIN_coeff_linear / 2.
    Returns a scalar because the channel filter will be applied in the main function.
    """
    # Convert from dB/Hz to linear scale
    RIN_coeff_lin = 10**(RIN_coeff_dB / 10)
    k_RIN = RIN_coeff_lin / 2
    return k_RIN * P_TX_sq_avg

def calc_S_ADC(PAPR, sigma_x_sq, ENOB, fs, f_array):
    """
    Calculates the PSD of ADC quantization noise.
    Based on the quantization noise model.
    """
    S_ADC = ((PAPR**-2) * sigma_x_sq / (12 * fs) * (2**(2 * ENOB - 2)))
    return np.full_like(f_array, S_ADC)

def calc_S_N(S_RIN_scalar, S_shot_array, S_th_array, S_ADC_array, H_ch_sq_array):
    """
    Consolidate the total noise S_N(f) at the equalizer input (Eq. 8).
    S_N(f) = S_RIN * |H_ch(f)|^2 + S_shot(f) + S_th(f) + S_ADC(f)
    """

    S_N_array = (S_RIN_scalar * H_ch_sq_array) + S_shot_array + S_th_array + S_ADC_array

    return S_N_array

