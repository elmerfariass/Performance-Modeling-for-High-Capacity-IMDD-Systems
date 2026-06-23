import numpy as np
from scipy.constants import k as kB  # Boltzmann constant
from scipy.special import erfc
# Physical constant
q = 1.602176634e-19 # Electron charge [C]


"""
Python Archive that contains all the functions used in the project.

Authors:
- [@jezraelP] Jezrael Pereira Filgueiras
- [@elmerfariass] Elmer Pimentel Farias
- [@JoaohMorais2] João Henrique Morais do Nascimento



Functions:

    - superGauss(f, n, B_3dB): Obtains the frequency response of a supergaussian filter
    - SNR_Folding(SNR, f, mu, Rs): Folds the SNR  on a bandwith equal to the symbol rate.
    - SNR_DFE(SNR_folded, f, Rs): Calculates the SNR at the output of a Decision Feedback Equalizer (DFE) based on the folded SNR.
    - SNR_FFE(f, Rs, SNR_Folding): Calculates the SNR at the output of a Feed Forward Equalizer (FFE) based on the folded SNR.
"""

def superGauss(f, n, B_3dB):
    '''
    Calculates the supergaussian filter frequency response for a given order n and 3dB bandwidth B_3dB.

    param:
        f (np.array): Frequency array (in Hz) for which to calculate the filter response.
        n (int): Order of the supergaussian filter.
        B_3dB (float): 3dB bandwidth of the filter (in Hz).
    
    return:
        H_sg (np.array): Frequency response of the supergaussian filter at the given frequencies.
    '''

    alpha = np.log(2)/2
    H_sg = np.exp(-alpha * (f / B_3dB)**(2*n))
    return H_sg

def SNR_Folding(SNR, f, mu, Rs):
    """
    Folds the SNR  on a bandwith equal to the symbol rate.

    param:
        SNR (np.array): SNR vector linear
        f (np.array): Frequency vector
        mu (int): Number of foldings
        Rs (float): Symbol rate [Baud]
    returns:
        SNR_folded (np.array): Folded SNR vector linear
    """

    df = f[1]-f[0]
    SNR_folded = np.zeros_like(SNR)
    shift = int(round(Rs/df))

    for i in range(-mu, mu+1):
        SNR_folded += np.roll(SNR, i*shift)
    return SNR_folded

def SNR_DFE(SNR_folded, f, Rs):
    """
    calculates the SNR at the output of a Decision Feedback Equalizer (DFE) based on the folded SNR.

    param:
        SNR_Folded (np.array): Folded SNR vector linear
        f (np.array): Frequency vector
        Rs (float): Symbol rate [Baud]
    returns:
        SNR_DFE (np.array): SNR at the output of the DFE
    """

    lim_inf = -Rs/2
    lim_sup = Rs/2
    T = 1/Rs

    idx = np.where((f >= lim_inf) & (f <= lim_sup))
    SNR_DFE = np.exp((T*np.trapezoid(np.log(SNR_folded[idx]+1), f[idx]))) - 1
    return SNR_DFE

def SNR_FFE(SNR_Folded, f, Rs):
    '''
    Calculates the SNR at the output of a Feed-Forward Equalizer (FFE) based on the folded SNR.

    param:
        f (np.array): Frequency vector
        Rs (float): Symbol rate [Baud]
        SNR_Folded (np.array): Folded SNR vector linear
    returns:
        SNR_FFE (float): SNR at the output of the FFE
    
    '''
    
    df = f[1] - f[0]
    T = 1.0 / Rs  # Período do símbolo [s]

    lim = Rs / 2.0

    mask = (f >= -lim) & (f <= lim)

    f_lim = f[mask]
    SNR_Folded_lim = SNR_Folded[mask]

    integrando = 1.0 / (SNR_Folded_lim + 1.0)
    integral = np.trapz(integrando, dx=df)

    SNR_FFE = (1.0 / (T * integral)) - 1.0

    return SNR_FFE



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


def calc_S_shot(P_RX_avg, G, F, R, fq_array):
    """
    Calculates the PSD of shot noise.
    Assuming k_shot = (G^2 * F * q )/ R.
    """
    k_shot = (G**2 * F * q) / R
    S_shot = k_shot * P_RX_avg
    return np.full_like(fq_array, S_shot)

def calc_S_shot_opticompy(P_RX_avg, R, Id, fq_array):
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
    - P_RX_avg: Average received optical power (W)
    - R: Photodiode responsivity (A/W)
    - Id: Dark current (A)
    - fq_array: Frequency array for the PSD (double-sided)
    """
    
    # Calculate the average photocurrent based on received power
    I_avg = P_RX_avg * R
    
    # Calculate the double-sided PSD matching OpticommPy's exact formula
    # S_shot = q * (photocurrent + dark_current)
    # (G and F are inherently 1 here, representing a PIN detector)
    S_shot = q * (I_avg + Id)
    
    #Return the constant PSD array across all frequencies
    return np.full_like(fq_array, S_shot)

def calc_S_RIN(P_TX_sq_avg, RIN_coeff, fq_array):
    """
    Calculates the PSD of RIN at the source.
    k_RIN = RIN_coeff_linear / 2.
    Returns a scalar, as the channel filter will be applied in the main function.

    Parameters:
    - P_TX_sq_avg: Average transmitted optical power squared (W^2)
    - RIN_coeff: Relative Intensity Noise coefficient (linear scale)
    - fq_array: Frequency array for the PSD (double-sided)

    Returns:
    - S_RIN: PSD of RIN (W^2/Hz), constant across all frequencies in fq_array

    """
    
    k_RIN = RIN_coeff / 2
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

    Parameters:
    - PAPR: Peak-to-Average Power Ratio of the signal
    - sigma_x_sq: Variance of the input signal to the ADC
    - ENOB: Effective Number of Bits of the ADC
    - fs: Sampling frequency of the ADC (Hz)
    - fq_array: Frequency array for the PSD (double-sided)

    Returns:
    - S_ADC: PSD of ADC quantization noise (W^2/Hz), constant across all frequencies in fq_array

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

    Returns:
    - S_ADC: PSD of ADC quantization noise (V^2/Hz), constant across all frequencies in fq_array

    """
    
    #  Find the peak voltage from the OpticommPy full-scale range
    Vpeak = (Vmax - Vmin) / 2
    
    #  The equation's numerator (PAPR * sigma_x^2) is mathematically equal to Vpeak^2
    PAPR_sigma_x_sq = Vpeak**2
    
    # Apply the exact equation as presented in the article/image
    # S_ADC(f) = (PAPR * sigma_x^2) / (12 * fs * 2^(2*ENOB - 2))
    S_ADC = PAPR_sigma_x_sq / (12 * outFs * (2**(2 * ENOB - 2)))
    
    return np.full_like(fq_array, S_ADC)

def calc_S_N(S_RIN_scalar, S_shot_array, S_th_array, S_ADC_array, H_ch_array, enable_S_ADC=True):
    """
    Consolidate the total noise S_N(f) at the equalizer input (Eq. 8).
    S_N(f) = S_RIN * |H_ch(f)|^2 + S_shot(f) + S_th(f) + S_ADC(f)
    
    Parameters:
    ...
    enable_S_ADC: bool, turns the S_ADC contribution on (True) or off (False).
    """
    
    # Define se a parcela do ADC entra no cálculo ou vira 0
    S_ADC_term = S_ADC_array if enable_S_ADC else 0

    S_N_array = (S_RIN_scalar * np.abs(H_ch_array ** 2)) + S_shot_array + S_th_array + S_ADC_term

    return S_N_array

def calculate_oma_outer(p_tx_avg_w, er_db):
    """
    Calculates the Outer OMA (Equation 7).

    Parameters:
    p_tx_avg_w (float): Average transmitted optical power [W].
    er_db (float): Extinction Ratio [dB].

    Returns:
    float: Outer OMA [W].
    """
    er_lin = 10**(er_db / 10.0)  # Conversão de dB para escala linear

    # Cálculo da OMA outer usando a equação(7)
    
    oma_outer = 2 * p_tx_avg_w * (er_lin - 1) / (er_lin + 1)
    return oma_outer

def get_spectral_snr(f, oma_outer, symbol_rate, h_t_f, h_ch_f, s_n_f):
    """
    Calcula o SNR espectral baseado na Equação (10)

    parâmetros:
    f: Frequência [Hz]
    oma_outer: OMA externa [W]
    symbol_rate: Taxa de símbolos [Hz]
    h_t_f: Resposta em frequência do transmissor [adimensional]
    h_ch_f: Resposta em frequência do canal [adimensional]
    s_n_f: Densidade espectral de potência do ruído [W^2/Hz]
    retorna:
    snr_f: SNR espectral [adimensional]
    """
    t = 1 / symbol_rate  # Período do símbolo [cite: 89]
    
    # Numerador: 
    numerator = (5/36) * t * (oma_outer**2) * (np.abs(h_t_f)**2) * (np.abs(h_ch_f)**2)
    
    # SNR(f) = Numerador / S_N(f) [cite: 125]
    snr_f = numerator / s_n_f
    return snr_f

def ber_from_snr_m_pam(snr_linear, M):
    """
    Calcula a BER aproximada para M-PAM a partir da SNR em escala linear.

    Fórmula do artigo:
        BER ≈ (M - 1)/(M log2(M)) * erfc(
            sqrt(3*SNR / (2*(M^2 - 1)))
        )

    Parâmetros
    ----------
    snr_linear : float ou np.ndarray
        SNR em escala linear, não em dB.

    M : int
        Ordem da modulação PAM. Ex: M=4 para 4-PAM.

    Retorna
    -------
    ber : float ou np.ndarray
        Bit Error Rate estimada.
    """
    snr_linear = np.asarray(snr_linear)

    ber = ((M - 1) / (M * np.log2(M))) * erfc(
        np.sqrt((3 * snr_linear) / (2 * (M**2 - 1)))
    )

    return ber
