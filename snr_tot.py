import numpy as np


def get_spectral_snr(f, oma_outer, symbol_rate, h_t_f, h_ch_f, s_n_f):
    """
    Calcula o SNR espectral baseado na Equação (10)

    parâmetros:
    f: Frequência [Hz]
    oma_outer: OMA externa [mW]
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