import numpy as np
from scipy.constants import k as kB  # Boltzmann constant
from scipy.special import erfc
from optic.comm.metrics import fastBERcalc
import scipy.special as sp
from scipy.constants import c
from optic.dsp.equalization import dfe, ffe



# Physical constant
q = 1.602176634e-19 # Electron charge [C]


"""
Python Archive that contains all the functions used in the project.

Authors:
- [@jezraelP] Jezrael Pereira Filgueiras
- [@elmerfariass] Elmer Pimentel Farias
- [@EduardoHFC] Eduardo Henrique Freitas Coura



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

def SNR_DFE(SNR_folded, SIM_N, f, Rs):
    """
    calculates the SNR at the output of a Decision Feedback Equalizer (DFE) based on the folded SNR.

    param:
        SNR_Folded (np.array): Folded SNR vector linear
        f (np.array): Frequency vector
        Rs (float): Symbol rate [Baud]
    returns:
        SNR_DFE (np.array): SNR at the output of the DFE
    """
    T = 1.0 / Rs  # Período do símbolo [s]
    f_int = np.linspace(-1 / (2 * T), 1 / (2 * T), SIM_N)
    snr_min = np.maximum(np.abs(SNR_folded), 1e-12)
    SNR_f_interp = 10**(np.interp(f_int, f, 10 * np.log10(snr_min)) / 10)

    snr_dfe_lin = np.exp(T * np.trapezoid(np.log(SNR_f_interp + 1), x=f_int)) - 1

    
    return 10*np.log10(snr_dfe_lin)

def SNR_FFE(SNR_Folded, SIM_N, f, Rs):
    '''
    Calculates the SNR at the output of a Feed-Forward Equalizer (FFE) based on the folded SNR.

    param:
        f (np.array): Frequency vector
        Rs (float): Symbol rate [Baud]
        SNR_Folded (np.array): Folded SNR vector linear
    returns:
        SNR_FFE (float): SNR at the output of the FFE
    
    '''
    
    T = 1.0 / Rs  # Período do símbolo [s]
    f_int = np.linspace(-1 / (2 * T), 1 / (2 * T), SIM_N)
    snr_min = np.maximum(np.abs(SNR_Folded), 1e-12)
    SNR_f_interp = 10**(np.interp(f_int, f, 10 * np.log10(snr_min)) / 10)
        
    snr_ffe_lin = 1 / (T * np.trapezoid(1 / (SNR_f_interp + 1), x=f_int)) - 1

    return 10*np.log10(snr_ffe_lin)



def calc_S_th(N_0, fq_array):
    """
    Calculates the PSD of thermal noise, assumed to be constant across frequency.
    Based on S_th = N_0 / 2.
    """
    return np.full_like(fq_array, N_0 / 2)



def calc_S_shot(P_RX, G, F, R, fq_array):
    """
    Calculates the PSD of shot noise.
    Assuming k_shot = (G^2 * F * q )/ R.
    """
    k_shot = (G**2 * F * q) / R
    S_shot = k_shot * P_RX
    return np.full_like(fq_array, S_shot)



def calc_S_RIN(Ps2, RIN_coeff_dB, fq_array ):
    """
    Calculates the PSD of RIN at the source.
    k_RIN = RIN_coeff_linear / 2.
    Returns a scalar, as the channel filter will be applied in the main function.
    """
    # Convert from dB/Hz to linear scale
    RIN_coeff_lin = 10**(RIN_coeff_dB / 10)
    k_RIN = RIN_coeff_lin / 2
    return np.full_like(fq_array, k_RIN * np.mean(Ps2**2))



def calc_S_ADC(PAPR, sigma_x_sq, ENOB, fs, fq_array):
    """
    Calculates the PSD of ADC quantization noise.
    Based on the quantization noise model.
    """
    S_ADC = ((PAPR) * sigma_x_sq / (12 * fs) * (2**(2 * ENOB - 2)))
    return np.full_like(fq_array, S_ADC)


def calc_S_N(S_RIN_scalar, S_shot_array, S_th_array, S_ADC_array, H_ch_sq_array, enable_S_ADC=True):
    """
    Consolidate the total noise S_N(f) at the equalizer input (Eq. 8).
    S_N(f) = S_RIN * |H_ch(f)|^2 + S_shot(f) + S_th(f) + S_ADC(f)
    
    Parameters:
    ...
    enable_S_ADC: bool, turns the S_ADC contribution on (True) or off (False).
    """
    
    # Define se a parcela do ADC entra no cálculo ou vira 0
    S_ADC_term = S_ADC_array if enable_S_ADC else 0

    S_N_array = (S_RIN_scalar * H_ch_sq_array) + S_shot_array + S_th_array + S_ADC_term

    return S_N_array

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

    ber = ((M - 1) / (M * np.log2(M))) * erfc(np.sqrt((3 * snr_linear) / (2 * (M**2 - 1))))

    return ber

def snr_to_ber_pam4(snr_lin):
    """Converte SNR linear elétrica em BER teórica para 4-PAM"""

    ber= (3/8) * sp.erfc(np.sqrt(snr_lin / 10))

    return ber

def calculate_ps2(p_tx_avg_w, M, OMA_outer):
    """
    Calculates the power levels for each PAM level.

    Parameters:
    p_tx_avg_w (float): Average transmitted optical power [W].
    M (int): Order of the PAM modulation.
    OMA_outer (float): Outer OMA [W].

    Returns:
    numpy.ndarray: Array of power levels for each PAM level [W].
    """

    niveis_pam = np.linspace(-1, 1, M)
    Ps2 = p_tx_avg_w + niveis_pam * (OMA_outer / 2)
    return Ps2

import numpy as np

def calc_S_ADC(Vpp, ENOB, Rs, fq_array, fs):
    """
    Calcula a Densidade Espectral de Potência (PSD) do ruído de quantização do ADC
    baseado na aproximação AWGN sobre a banda de Nyquist.
    
    Parâmetros:
    - Vpp: Tensão pico a pico na entrada do ADC (Full-scale range).
    - ENOB: Effective Number of Bits (Número Efetivo de Bits) do ADC.
    - Rs: Symbol rate (ou Taxa de Amostragem, dependendo da convenção matemática do modelo).
    - fq_array: Array de frequências.
    
    Retorna:
    - Array com o valor constante da PSD do ruído de quantização para todas as frequências.
    """
    
    # 1. Calcula o passo de quantização (Delta)
    Delta = Vpp / (2**ENOB)
    
    # 2. Calcula a variância teórica do ruído de quantização
    sigma_q2 = (Delta**2) / 12.0
    
    # 3. Calcula o S_ADC (E[|n_q(f)|^2]) isolando-o na fórmula da imagem
    S_ADC_scalar = sigma_q2 / (fs/2)
    
    # 4. Retorna como um array do mesmo tamanho que fq_array (espectro plano / branco)
    return np.full_like(fq_array, S_ADC_scalar)



def calculate_oma_outer(p_tx_average, er_dB):
    """Calcula a Outer OMA a partir da potência média e da ER."""
    er_linear = 10**(er_dB / 10)
    return 2 * p_tx_average * (er_linear - 1) / (er_linear + 1)


def normalize_affine(received, reference):
    """Ajusta ganho e offset da saída do equalizador."""
    received = np.real(np.asarray(received)).reshape(-1)
    reference = np.real(np.asarray(reference)).reshape(-1)

    common_length = min(len(received), len(reference))
    received = received[:common_length]
    reference = reference[:common_length]

    regression_matrix = np.column_stack((received, np.ones_like(received)))
    gain, offset = np.linalg.lstsq(regression_matrix, reference, rcond=None)[0]

    return gain * received + offset



def evaluate_equalizer_output(equalized_signal, reference_symbols, guard_symbols, M):
    """Remove bordas, normaliza e calcula BER, SER e SNR."""

    equalized_signal = np.real(np.asarray(equalized_signal)).reshape(-1)
    reference_symbols = np.real(np.asarray(reference_symbols)).reshape(-1)

    common_length = min(len(equalized_signal), len(reference_symbols))
    equalized_signal = equalized_signal[:common_length]
    reference_symbols = reference_symbols[:common_length]

    if common_length <= 2 * guard_symbols:
        raise RuntimeError("A sequência é muito curta para o descarte escolhido.")

    equalized_eval = equalized_signal[guard_symbols:common_length - guard_symbols]
    reference_eval = reference_symbols[guard_symbols:common_length - guard_symbols]

    equalized_eval = normalize_affine(equalized_eval, reference_eval)

    BER, SER, SNR_dB = fastBERcalc(equalized_eval, reference_eval, M, "pam")

    BER = float(np.asarray(BER).reshape(-1)[0])
    SER = float(np.asarray(SER).reshape(-1)[0])
    SNR_dB = float(np.asarray(SNR_dB).reshape(-1)[0])

    return BER, SER, SNR_dB

def evaluate_equalized_output(equalized_signal, reference_symbols, guard_symbols, M):
    """
    Calcula a SNR e a BER após o FFE.
    """

    equalized_signal = np.real(np.asarray(equalized_signal)).reshape(-1)
    reference_symbols = np.real(np.asarray(reference_symbols)).reshape(-1)

    common_length = min(len(equalized_signal), len(reference_symbols))

    equalized_signal = equalized_signal[:common_length]
    reference_symbols = reference_symbols[:common_length]

    if common_length <= 2 * guard_symbols:
        raise RuntimeError("Sequência insuficiente para o descarte.")

    equalized_evaluation = equalized_signal[
        guard_symbols:common_length - guard_symbols
    ]

    reference_evaluation = reference_symbols[
        guard_symbols:common_length - guard_symbols
    ]

    equalized_evaluation = normalize_affine(
        equalized_evaluation,
        reference_evaluation
    )

    signal_power = np.mean(reference_evaluation**2)
    error_power = np.mean(
        (reference_evaluation - equalized_evaluation)**2
    )

    snr_linear = signal_power / max(
        error_power,
        np.finfo(float).tiny
    )

    snr_dB = 10 * np.log10(snr_linear)

    BER, SER, _ = fastBERcalc(
        equalized_evaluation,
        reference_evaluation,
        M,
        "pam"
    )

    BER = float(np.asarray(BER).reshape(-1)[0])

    evaluated_bits = len(reference_evaluation) * int(np.log2(M))

    # Limite de representação quando nenhum erro é observado
    if BER <= 0:
        BER_plot = 0.5 / evaluated_bits
    else:
        BER_plot = BER

    return snr_dB, BER_plot



def calculate_output_snr(equalized_signal, reference_symbols, guard_symbols):
    """Calcula a SNR a partir da potência do sinal e do MSE."""

    equalized_signal = np.real(np.asarray(equalized_signal)).reshape(-1)
    reference_symbols = np.real(np.asarray(reference_symbols)).reshape(-1)

    common_length = min(len(equalized_signal), len(reference_symbols))
    equalized_signal = equalized_signal[:common_length]
    reference_symbols = reference_symbols[:common_length]

    if common_length <= 2 * guard_symbols:
        raise RuntimeError("Sequência insuficiente para o descarte escolhido.")

    equalized_evaluation = equalized_signal[guard_symbols:common_length - guard_symbols]
    reference_evaluation = reference_symbols[guard_symbols:common_length - guard_symbols]

    equalized_evaluation = normalize_affine(equalized_evaluation, reference_evaluation)

    signal_power = np.mean(reference_evaluation**2)
    error_power = np.mean((reference_evaluation - equalized_evaluation)**2)
    snr_linear = signal_power / max(error_power, np.finfo(float).tiny)

    return 10 * np.log10(snr_linear)


def fold_spectral_snr(spectral_snr, frequencies, Rs, number_of_folds):
    """Realiza o dobramento espectral sem enrolamento circular."""

    spectral_snr = np.asarray(spectral_snr, dtype=float)
    frequencies = np.asarray(frequencies, dtype=float)

    sorting_indexes = np.argsort(frequencies)
    frequency_sorted = frequencies[sorting_indexes]
    snr_sorted = spectral_snr[sorting_indexes]

    folded_sorted = np.zeros_like(snr_sorted)

    for folding_index in range(-number_of_folds, number_of_folds + 1):
        shifted_frequency = frequency_sorted - folding_index * Rs
        folded_sorted += np.interp(shifted_frequency, frequency_sorted, snr_sorted, left=0.0, right=0.0)

    folded = np.empty_like(folded_sorted)
    folded[sorting_indexes] = folded_sorted

    return folded


def analytical_ffe_snr(frequencies, spectral_snr, Rs):
    """Calcula a SNR na saída de um FFE MMSE ideal."""

    nyquist_mask = (frequencies >= -Rs / 2) & (frequencies <= Rs / 2)
    frequency_nyquist = frequencies[nyquist_mask]
    snr_nyquist = spectral_snr[nyquist_mask]

    symbol_period = 1 / Rs
    integral = np.trapezoid(1 / (1 + snr_nyquist), x=frequency_nyquist)

    snr_linear = 1 / (symbol_period * integral) - 1
    snr_linear = max(float(snr_linear), np.finfo(float).tiny)

    return 10 * np.log10(snr_linear)

def chromatic_dispersion_small_signal(frequencies, dispersion_ps_nm_km, length_km, wavelength_m):
    """Resposta elétrica de pequeno sinal da dispersão cromática."""

    frequencies = np.asarray(frequencies, dtype=float)

    if length_km == 0:
        return np.ones_like(frequencies)

    dispersion_SI = dispersion_ps_nm_km * 1e-6
    length_m = length_km * 1e3
    carrier_frequency = c / wavelength_m

    argument = np.pi * c * dispersion_SI * length_m * (frequencies / carrier_frequency)**2

    return np.cos(argument)


def chromatic_dispersion_field_response(frequencies, dispersion_ps_nm_km, length_km, wavelength_m):
    """Resposta da dispersão aplicada ao campo óptico."""

    frequencies = np.asarray(frequencies, dtype=float)

    if length_km == 0:
        return np.ones_like(frequencies, dtype=complex)

    dispersion_SI = dispersion_ps_nm_km * 1e-6
    length_m = length_km * 1e3
    carrier_frequency = c / wavelength_m

    phase = np.pi * c * dispersion_SI * length_m * (frequencies / carrier_frequency)**2

    return np.exp(1j * phase)

def apply_frequency_response(signal, frequency_response):
    """Aplica uma resposta em frequência por meio da FFT."""

    signal = np.asarray(signal).reshape(-1)
    frequency_response = np.asarray(frequency_response).reshape(-1)

    return np.fft.ifft(np.fft.fft(signal) * frequency_response)


def calculate_analytical_point(
    received_power_W,
    extinction_ratio_dB,
    fiber_length_km,
    analytical_frequencies,
    transmitter_response,
    bandwidth_response,
    dispersion_coefficient,
    wavelength,
    symbol_rate,
    APD_gain,
    APD_excess_factor,
    responsivity,
    RIN_linear,
    thermal_current_PSD,
    electron_charge,
    M
):
    """Calcula a SNR global e a BER média dos olhos para um ponto."""

    symbol_period = 1 / symbol_rate
    OMA_received = calculate_oma_outer(received_power_W, extinction_ratio_dB)

    dispersion_response = chromatic_dispersion_small_signal(
        analytical_frequencies,
        dispersion_coefficient,
        fiber_length_km,
        wavelength
    )

    channel_response = bandwidth_response * dispersion_response
    channel_power_response = np.abs(channel_response)**2

    signal_current_PSD = (
        (APD_gain * responsivity)**2
        * (5 / 36)
        * symbol_period
        * OMA_received**2
        * np.abs(transmitter_response)**2
        * channel_power_response
    )

    received_levels = received_power_W + np.linspace(-1, 1, M) * OMA_received / 2

    mean_square_received_power = np.mean(received_levels**2)

    RIN_current_PSD = (
        (APD_gain * responsivity)**2
        * RIN_linear
        * mean_square_received_power
        / 2
        * channel_power_response
    )

    shot_current_PSD = APD_gain**2 * APD_excess_factor * electron_charge * responsivity * received_power_W
    thermal_PSD = thermal_current_PSD / 2

    total_noise_PSD = RIN_current_PSD + shot_current_PSD + thermal_PSD

    spectral_snr = signal_current_PSD / np.maximum(total_noise_PSD, np.finfo(float).tiny)
    spectral_snr = np.nan_to_num(spectral_snr, nan=0.0, posinf=0.0, neginf=0.0)

    folded_snr = fold_spectral_snr(spectral_snr, analytical_frequencies, symbol_rate, number_of_folds=4)
    global_snr_linear = analytical_ffe_snr(analytical_frequencies, folded_snr, symbol_rate)

    global_snr_dB = 10 * np.log10(max(global_snr_linear, np.finfo(float).tiny))

    eye_BER_values = []

    for eye_index in range(M - 1):

        eye_levels = received_levels[eye_index:eye_index + 2]

        eye_average_power = np.mean(eye_levels)
        eye_mean_square_power = np.mean(eye_levels**2)

        eye_RIN_PSD = (
            (APD_gain * responsivity)**2
            * RIN_linear
            * eye_mean_square_power
            / 2
            * channel_power_response
        )

        eye_shot_PSD = APD_gain**2 * APD_excess_factor * electron_charge * responsivity * eye_average_power
        eye_total_noise_PSD = eye_RIN_PSD + eye_shot_PSD + thermal_PSD

        eye_spectral_snr = signal_current_PSD / np.maximum(eye_total_noise_PSD, np.finfo(float).tiny)

        eye_folded_snr = fold_spectral_snr(
            eye_spectral_snr,
            analytical_frequencies,
            symbol_rate,
            number_of_folds=4
        )

        eye_snr_linear = analytical_ffe_snr(analytical_frequencies, eye_folded_snr, symbol_rate)
        eye_BER = ber_from_snr_m_pam(eye_snr_linear, M)

        eye_BER_values.append(float(eye_BER))

    global_BER = np.mean(eye_BER_values)

    return global_snr_dB, global_BER


def fold_snr_uniform_grid(spectral_snr, frequency_step, symbol_rate, sampling_frequency):
    """Realiza o dobramento espectral em uma grade uniforme."""

    spectral_snr = np.asarray(spectral_snr, dtype=float)

    shift_samples = int(np.round(symbol_rate / frequency_step))
    number_of_folds = int(np.ceil(sampling_frequency / (2 * symbol_rate))) + 1

    folded_snr = np.zeros_like(spectral_snr)
    number_samples = len(spectral_snr)

    for folding_index in range(-number_of_folds, number_of_folds + 1):

        shift = folding_index * shift_samples

        if shift == 0:
            folded_snr += spectral_snr

        elif shift > 0 and shift < number_samples:
            folded_snr[shift:] += spectral_snr[:number_samples - shift]

        elif shift < 0:
            shift_abs = -shift

            if shift_abs < number_samples:
                folded_snr[:number_samples - shift_abs] += spectral_snr[shift_abs:]

    return folded_snr

def equalizer_output_snr(frequencies, folded_spectral_snr, symbol_rate, equalizer_type):
    frequencies = np.asarray(frequencies, dtype=float)
    folded_spectral_snr = np.asarray(folded_spectral_snr, dtype=float)

    nyquist_mask = (frequencies >= -symbol_rate / 2) & (frequencies <= symbol_rate / 2)
    frequency_nyquist = frequencies[nyquist_mask]
    snr_nyquist = folded_spectral_snr[nyquist_mask]

    symbol_period = 1 / symbol_rate
    equalizer_type = equalizer_type.upper()

    if equalizer_type == "FFE":
        integral = np.trapezoid(1 / (1 + snr_nyquist), x=frequency_nyquist)
        output_snr = 1 / (symbol_period * integral) - 1

    elif equalizer_type == "DFE":
        integral = np.trapezoid(np.log1p(np.maximum(snr_nyquist, 0.0)), x=frequency_nyquist)
        output_snr = np.exp(symbol_period * integral) - 1

    else:
        raise ValueError("equalizer_type deve ser 'FFE' ou 'DFE'.")

    return max(float(output_snr), 0.0)


def dbm_to_w(power_dBm):
    """Converte potência de dBm para watts."""
    return 1e-3 * 10**(np.asarray(power_dBm, dtype=float) / 10)

def pam_levels(M):
    """Retorna os níveis naturais de uma constelação M-PAM."""
    return np.arange(-(M - 1), M, 2, dtype=float)

def fold_snr_uniform_grid(spectral_snr, frequency_step, symbol_rate, sampling_frequency):
    """Realiza o dobramento espectral em uma grade uniforme."""

    spectral_snr = np.asarray(spectral_snr, dtype=float)

    shift_samples = int(np.round(symbol_rate / frequency_step))
    number_of_folds = int(np.ceil(sampling_frequency / (2 * symbol_rate))) + 1

    folded_snr = np.zeros_like(spectral_snr)
    number_samples = len(spectral_snr)

    for folding_index in range(-number_of_folds, number_of_folds + 1):

        shift = folding_index * shift_samples

        if shift == 0:
            folded_snr += spectral_snr

        elif shift > 0 and shift < number_samples:
            folded_snr[shift:] += spectral_snr[:number_samples - shift]

        elif shift < 0:
            shift_abs = -shift

            if shift_abs < number_samples:
                folded_snr[:number_samples - shift_abs] += spectral_snr[shift_abs:]

    return folded_snr


