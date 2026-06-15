import numpy as np
"""
Python Archive that contains all the functions used in the project.

Authors:
- [@jezraelP] Jezrael Pereira Filgueiras



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