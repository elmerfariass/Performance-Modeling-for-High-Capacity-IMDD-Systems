import numpy as np

def calculate_oma_outer(p_tx_avg_mw, er_db):
    """
    Calculates the Outer OMA (Equation 7).

    Parameters:
    p_tx_avg_mw (float): Average transmitted optical power [mW].
    er_db (float): Extinction Ratio [dB].

    Returns:
    float: Outer OMA [mW].
    """
    er_lin = 10**(er_db / 10.0)  # Conversão de dB para escala linear

    # Cálculo da OMA outer usando a equação(7)
    
    oma_outer = 2 * p_tx_avg_mw * (er_lin - 1) / (er_lin + 1)
    return oma_outer

