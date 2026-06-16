import numpy as np

def calculate_oma_outer(p_tx_avg_mw, er_db):
    """
    Calcula a OMA outer (Equação 7)

    recebe a potência média do transmissor em mW e a razão de extinção em dB, e retorna a OMA outer em mW.
    """
    er_lin = 10**(er_db / 10.0)  # Conversão de dB para escala linear

    # Cálculo da OMA outer usando a equação(7)
    
    oma_outer = 2 * p_tx_avg_mw * (er_lin - 1) / (er_lin + 1)
    return oma_outer

