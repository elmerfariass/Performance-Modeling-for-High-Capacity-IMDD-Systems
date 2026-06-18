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

def snr_equalizer_output_ffe(snr_folded, freqs_base, T):
    integrand = 1 / (snr_folded + 1)
    integral = np.trapz(integrand, freqs_base)

    return (1 / T) * (1 / integral) - 1


def snr_equalizer_output_dfe(snr_folded, freqs_base, T):
    integrand = np.log(snr_folded + 1)
    integral = np.trapz(integrand, freqs_base)

    return np.exp(T * integral) - 1