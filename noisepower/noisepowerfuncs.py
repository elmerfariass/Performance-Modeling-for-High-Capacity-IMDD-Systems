import numpy as np

# Constante física
q = 1.602176634e-19 # Carga do elétron [C]

def calc_S_th(N_0, f_array):
    """
    Calcula a PSD do ruído térmico, assumido como constante na frequência.
    Baseado em S_th = N_0 / 2.
    """
    return np.full_like(f_array, N_0 / 2)

def calc_S_shot(P_RX, G, F, R, f_array):
    """
    Calcula a PSD do ruído de disparo (shot noise).
    Assumindo k_shot = G^2 * F * q / R.
    """
    k_shot = (G**2 * F * q) / R
    S_shot_val = k_shot * P_RX
    return np.full_like(f_array, S_shot_val)

def calc_S_RIN(P_TX_sq_avg, RIN_coeff_dB):
    """
    Calcula a PSD do RIN na fonte. 
    k_RIN = RIN_coeff_linear / 2.
    Retorna um escalar, pois o filtro do canal será aplicado na função principal.
    """
    # Converte de dB/Hz para escala linear
    RIN_coeff_lin = 10**(RIN_coeff_dB / 10)
    k_RIN = RIN_coeff_lin / 2
    return k_RIN * P_TX_sq_avg

def calc_S_ADC(PAPR, sigma_x_sq, ENOB, f_s, f_array):
    """
    Calcula a PSD do ruído de quantização do ADC.
    Baseado na Equação (9) do artigo.
    """
    S_ADC_val = ((PAPR**-2) * sigma_x_sq * (2**(-2 * ENOB))) / (12 * f_s / 2)
    return np.full_like(f_array, S_ADC_val)

def calc_S_N(S_RIN_scalar, S_shot_array, S_th_array, S_ADC_array, H_ch_sq_array):
    """
    Consolida o ruído total S_N(f) na entrada do equalizador (Eq. 8).
    S_N(f) = S_RIN * |H_ch(f)|^2 + S_shot(f) + S_th(f) + S_ADC(f)
    """
    # Graças ao broadcasting do numpy, multiplicar o escalar S_RIN pelo 
    # array H_ch_sq_array gera o vetor correto na frequência.
    S_N_array = (S_RIN_scalar * H_ch_sq_array) + S_shot_array + S_th_array + S_ADC_array
    
    return S_N_array

