# Performance Modeling for High-Capacity IMDD Systems

This repository contains the source code and simulation framework developed as the final project for the Digital Signal Processing (DSP) training program at Virtus-CC, conducted under the supervision of Professor Edson Porto da Silva.

## Project Description

The primary objective of this project is to model the performance of High-Capacity Intensity Modulation and Direct Detection (IMDD) systems. The study focuses on reproducing the analytical results derived from the reference article "An Analytical Model for Performance Estimation in Modern High-Capacity IMDD Systems". 

The reference article proposes an analytical model to estimate the signal-to-noise ratio (SNR) and the Bit Error Rate (BER) at the output of a receiver adaptive equalizer in IMDD optical transmission systems. To validate these theoretical frameworks, our project performs a comparative analysis between the derived analytical models and numerical simulations obtained using `opticommpy`—an open-source Python library for optical communication systems authored by Professor Edson Porto da Silva.

## Objectives

* **Analytical Reproduction:** Derivation and implementation of the mathematical models for IMDD system performance as proposed in the reference literature.
* **Comparative Validation:** Benchmarking the analytical results against numerical simulations performed via the `opticommpy` framework to verify accuracy.
* **DSP Application:** Practical application of digital signal processing techniques to mitigate transmission constraints in high-capacity optical links.

## Technical Scope

Based on the referenced methodology, the simulation and analytical tools encompass the following parameters:

* **Modulation Analysis:** Support for M-PAM modulation formats, specifically focusing on 4-PAM solutions for short-reach systems.
* **Channel Modeling:** Implementation of optoelectronic bandwidth limitations and chromatic dispersion (CD) effects.
* **Noise Modeling:** Integration of relative intensity noise (RIN), shot noise, thermal noise, and quantization noise associated with the analog-to-digital converters.
* **Signal Processing:** Deployment of receiver equalization schemes, including feed-forward equalization (FFE) and decision-feedback equalization (DFE).

## References
[1] G. Rizzelli, P. Torres-Ferrera, F. Forghieri, and R. Gaudino, "An Analytical Model for Performance Estimation in Modern High-Capacity IMDD Systems," Journal of Lightwave Technology, vol. 42, no. 5, pp. 1443-1452, March 1, 2024.
