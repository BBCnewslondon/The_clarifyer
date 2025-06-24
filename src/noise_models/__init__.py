"""
Noise Models Module

This module provides various noise models for contaminating astrophysical signals.
"""

import numpy as np
from typing import Union, Tuple, Optional
import scipy.signal as signal
from scipy import stats


def add_gaussian_noise(
    clean_signal: np.ndarray,
    snr: float,
    noise_std: Optional[float] = None
) -> np.ndarray:
    """
    Add Gaussian white noise to a signal.
    
    Parameters:
    -----------
    clean_signal : np.ndarray
        Clean input signal
    snr : float
        Desired signal-to-noise ratio in dB
    noise_std : float, optional
        Standard deviation of noise. If None, calculated from SNR.
    
    Returns:
    --------
    np.ndarray
        Noisy signal
    """
    signal_power = np.mean(clean_signal**2)
    
    if noise_std is None:
        # Calculate noise standard deviation from desired SNR
        snr_linear = 10**(snr/10)
        noise_power = signal_power / snr_linear
        noise_std = np.sqrt(noise_power)
    
    noise = np.random.normal(0, noise_std, len(clean_signal))
    noisy_signal = clean_signal + noise
    
    return noisy_signal


def add_colored_noise(
    clean_signal: np.ndarray,
    snr: float,
    psd_model: str = "LIGO_design",
    sampling_rate: int = 4096
) -> np.ndarray:
    """
    Add colored noise with realistic power spectral density.
    
    Parameters:
    -----------
    clean_signal : np.ndarray
        Clean input signal
    snr : float
        Desired signal-to-noise ratio in dB
    psd_model : str
        PSD model ('LIGO_design', 'advanced_LIGO', 'pink', 'brown')
    sampling_rate : int
        Sampling rate in Hz
    
    Returns:
    --------
    np.ndarray
        Noisy signal with colored noise
    """
    N = len(clean_signal)
    freqs = np.fft.fftfreq(N, 1/sampling_rate)
    freqs[0] = 1e-10  # Avoid division by zero
    
    # Generate power spectral density
    if psd_model == "LIGO_design":
        psd = ligo_design_psd(np.abs(freqs))
    elif psd_model == "advanced_LIGO":
        psd = advanced_ligo_psd(np.abs(freqs))
    elif psd_model == "pink":
        psd = 1 / np.abs(freqs)
        psd[0] = psd[1]
    elif psd_model == "brown":
        psd = 1 / (np.abs(freqs)**2)
        psd[0] = psd[1]
    else:
        raise ValueError(f"Unknown PSD model: {psd_model}")
    
    # Generate white noise
    white_noise = np.random.normal(0, 1, N)
    
    # Color the noise
    noise_fft = np.fft.fft(white_noise)
    colored_noise_fft = noise_fft * np.sqrt(psd)
    colored_noise = np.real(np.fft.ifft(colored_noise_fft))
    
    # Scale to desired SNR
    signal_power = np.mean(clean_signal**2)
    noise_power = np.mean(colored_noise**2)
    snr_linear = 10**(snr/10)
    
    scaling_factor = np.sqrt(signal_power / (snr_linear * noise_power))
    colored_noise *= scaling_factor
    
    return clean_signal + colored_noise


def ligo_design_psd(freqs: np.ndarray) -> np.ndarray:
    """
    LIGO design sensitivity power spectral density.
    
    Parameters:
    -----------
    freqs : np.ndarray
        Frequency array in Hz
    
    Returns:
    --------
    np.ndarray
        Power spectral density in strain^2/Hz
    """
    # Simplified LIGO design PSD
    # Real LIGO PSD is more complex
    
    # Seismic wall at low frequencies
    seismic = (freqs / 10)**(-4)
    
    # Thermal noise
    thermal = 1e-46 * np.ones_like(freqs)
    
    # Shot noise at high frequencies
    shot = 1e-47 * (freqs / 100)**2
    
    # Combine noise sources
    psd = seismic + thermal + shot
    
    # High-pass filter effect
    psd[freqs < 10] *= 1e10
    
    return psd


def advanced_ligo_psd(freqs: np.ndarray) -> np.ndarray:
    """
    Advanced LIGO design sensitivity PSD.
    
    Parameters:
    -----------
    freqs : np.ndarray
        Frequency array in Hz
    
    Returns:
    --------
    np.ndarray
        Power spectral density in strain^2/Hz
    """
    # More accurate Advanced LIGO PSD model
    
    # Low frequency: seismic + suspension thermal
    low_freq = (freqs / 20)**(-4.14) * 1e-49
    
    # Mid frequency: coating thermal + substrate thermal
    mid_freq = 2.26e-47 * (1 + (freqs/55)**2)**(-1)
    
    # High frequency: quantum shot noise
    high_freq = 3.4e-49 * (freqs/1000)**2
    
    # Combine
    psd = low_freq + mid_freq + high_freq
    
    # Additional features
    # Violin mode resonances (simplified)
    violin_freqs = [500, 1000, 1500]  # Hz
    for vf in violin_freqs:
        violin_peak = 1e-48 / (1 + ((freqs - vf)/10)**2)
        psd += violin_peak
    
    return psd


def add_glitches(
    signal: np.ndarray,
    sampling_rate: int = 4096,
    glitch_rate: float = 0.1,
    glitch_types: list = None
) -> np.ndarray:
    """
    Add instrumental glitches to the signal.
    
    Parameters:
    -----------
    signal : np.ndarray
        Input signal
    sampling_rate : int
        Sampling rate in Hz
    glitch_rate : float
        Average glitches per second
    glitch_types : list
        Types of glitches to include
    
    Returns:
    --------
    np.ndarray
        Signal with glitches added
    """
    if glitch_types is None:
        glitch_types = ['blip', 'whistle', 'wandering_line']
    
    duration = len(signal) / sampling_rate
    num_glitches = np.random.poisson(glitch_rate * duration)
    
    noisy_signal = signal.copy()
    
    for _ in range(num_glitches):
        glitch_type = np.random.choice(glitch_types)
        glitch_time = np.random.uniform(0, duration)
        glitch_sample = int(glitch_time * sampling_rate)
        
        if glitch_type == 'blip':
            # Short duration transient
            glitch = generate_blip_glitch(sampling_rate)
        elif glitch_type == 'whistle':
            # Frequency evolving transient
            glitch = generate_whistle_glitch(sampling_rate)
        elif glitch_type == 'wandering_line':
            # Narrow-band artifact
            glitch = generate_wandering_line_glitch(sampling_rate, duration=0.5)
        
        # Add glitch to signal
        end_sample = min(glitch_sample + len(glitch), len(noisy_signal))
        glitch_length = end_sample - glitch_sample
        
        if glitch_length > 0:
            noisy_signal[glitch_sample:end_sample] += glitch[:glitch_length]
    
    return noisy_signal


def generate_blip_glitch(sampling_rate: int) -> np.ndarray:
    """Generate a blip glitch (short transient)."""
    duration = np.random.uniform(0.01, 0.1)  # 10-100 ms
    N = int(duration * sampling_rate)
    time = np.linspace(0, duration, N)
      # Damped sinusoid
    freq = np.random.uniform(50, 500)  # Hz
    amplitude = np.random.uniform(1e-21, 1e-19)
    decay_time = duration / 3
    
    glitch = amplitude * np.exp(-time/decay_time) * np.sin(2*np.pi*freq*time)
    
    # Apply window
    try:
        # Try scipy.signal.windows.tukey first (newer versions)
        from scipy.signal.windows import tukey
        window = tukey(N, alpha=0.5)
    except ImportError:
        try:
            # Fallback to scipy.signal.tukey (older versions)
            window = signal.tukey(N, alpha=0.5)
        except AttributeError:
            # Create a simple tapered window as fallback
            alpha = 0.5
            n_taper = int(alpha * N / 2)
            window = np.ones(N)
            # Taper the beginning
            window[:n_taper] = 0.5 * (1 + np.cos(np.pi * (np.arange(n_taper) / n_taper - 1)))
            # Taper the end
            window[-n_taper:] = 0.5 * (1 + np.cos(np.pi * np.arange(n_taper) / n_taper))
    
    glitch *= window
    
    return glitch


def generate_whistle_glitch(sampling_rate: int) -> np.ndarray:
    """Generate a whistle glitch (frequency evolving)."""
    duration = np.random.uniform(0.1, 1.0)  # 100ms - 1s
    N = int(duration * sampling_rate)
    time = np.linspace(0, duration, N)
    
    # Frequency evolution
    f0 = np.random.uniform(100, 300)  # Hz
    f1 = np.random.uniform(200, 800)  # Hz
    freq_evolution = f0 + (f1 - f0) * time / duration
      # Generate signal
    phase = 2 * np.pi * np.cumsum(freq_evolution) / sampling_rate
    amplitude = np.random.uniform(1e-21, 1e-20)
    
    glitch = amplitude * np.sin(phase)
    
    # Apply window
    try:
        # Try scipy.signal.windows.tukey first (newer versions)
        from scipy.signal.windows import tukey
        window = tukey(N, alpha=0.1)
    except ImportError:
        try:
            # Fallback to scipy.signal.tukey (older versions)
            window = signal.tukey(N, alpha=0.1)
        except AttributeError:
            # Create a simple tapered window as fallback
            alpha = 0.1
            n_taper = int(alpha * N / 2)
            window = np.ones(N)
            # Taper the beginning
            window[:n_taper] = 0.5 * (1 + np.cos(np.pi * (np.arange(n_taper) / n_taper - 1)))
            # Taper the end
            window[-n_taper:] = 0.5 * (1 + np.cos(np.pi * np.arange(n_taper) / n_taper))
    
    glitch *= window
    
    return glitch


def generate_wandering_line_glitch(sampling_rate: int, duration: float) -> np.ndarray:
    """Generate a wandering line glitch (narrow-band)."""
    N = int(duration * sampling_rate)
    time = np.linspace(0, duration, N)
    
    # Central frequency with random walk
    f_center = np.random.uniform(50, 500)  # Hz
    freq_noise = np.cumsum(np.random.normal(0, 0.1, N))  # Random walk
    freq = f_center + freq_noise
    
    # Generate signal
    phase = 2 * np.pi * np.cumsum(freq) / sampling_rate
    amplitude = np.random.uniform(1e-22, 1e-21)
    
    glitch = amplitude * np.sin(phase)
    
    return glitch


def add_line_noise(
    signal: np.ndarray,
    sampling_rate: int = 4096,
    line_frequencies: list = None,
    line_amplitudes: list = None
) -> np.ndarray:
    """
    Add power line harmonics and other narrow-band interference.
    
    Parameters:
    -----------
    signal : np.ndarray
        Input signal
    sampling_rate : int
        Sampling rate in Hz
    line_frequencies : list
        Frequencies of line noise in Hz
    line_amplitudes : list
        Amplitudes of line noise
    
    Returns:
    --------
    np.ndarray
        Signal with line noise added
    """
    if line_frequencies is None:
        # Common power line frequencies and harmonics
        line_frequencies = [60, 120, 180, 240, 300]  # Hz (US power grid)
    
    if line_amplitudes is None:
        line_amplitudes = [1e-21, 5e-22, 3e-22, 2e-22, 1e-22]
    
    duration = len(signal) / sampling_rate
    time = np.linspace(0, duration, len(signal))
    
    line_noise = np.zeros_like(signal)
    
    for freq, amp in zip(line_frequencies, line_amplitudes):
        # Add some frequency jitter
        freq_jitter = freq + np.random.normal(0, 0.1)
        
        # Add some amplitude modulation
        amp_modulation = 1 + 0.1 * np.sin(2*np.pi*0.1*time)  # 0.1 Hz modulation
        
        line_component = amp * amp_modulation * np.sin(2*np.pi*freq_jitter*time)
        line_noise += line_component
    
    return signal + line_noise


def calculate_snr(signal: np.ndarray, noise: np.ndarray) -> float:
    """
    Calculate signal-to-noise ratio.
    
    Parameters:
    -----------
    signal : np.ndarray
        Clean signal
    noise : np.ndarray
        Noise component
    
    Returns:
    --------
    float
        SNR in dB
    """
    signal_power = np.mean(signal**2)
    noise_power = np.mean(noise**2)
    
    if noise_power == 0:
        return np.inf
    
    snr_linear = signal_power / noise_power
    snr_db = 10 * np.log10(snr_linear)
    
    return snr_db


if __name__ == "__main__":
    # Example usage
    import matplotlib.pyplot as plt
    from signal_simulation import generate_gravitational_wave
    
    print("Generating example noisy signals...")
    
    # Generate clean signal
    time, clean_signal = generate_gravitational_wave(duration=1.0)
    
    # Add different types of noise
    gaussian_noisy = add_gaussian_noise(clean_signal, snr=10)
    colored_noisy = add_colored_noise(clean_signal, snr=10, psd_model="LIGO_design")
    glitch_noisy = add_glitches(clean_signal.copy())
    
    # Plot comparison
    fig, axes = plt.subplots(4, 1, figsize=(12, 10))
    
    axes[0].plot(time, clean_signal)
    axes[0].set_title('Clean Signal')
    axes[0].set_ylabel('Strain')
    axes[0].grid(True)
    
    axes[1].plot(time, gaussian_noisy)
    axes[1].set_title('Gaussian Noise (SNR=10 dB)')
    axes[1].set_ylabel('Strain')
    axes[1].grid(True)
    
    axes[2].plot(time, colored_noisy)
    axes[2].set_title('Colored Noise (LIGO PSD)')
    axes[2].set_ylabel('Strain')
    axes[2].grid(True)
    
    axes[3].plot(time, glitch_noisy)
    axes[3].set_title('With Glitches')
    axes[3].set_ylabel('Strain')
    axes[3].set_xlabel('Time (s)')
    axes[3].grid(True)
    
    plt.tight_layout()
    plt.savefig('noise_examples.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Noise examples generated and saved as 'noise_examples.png'")
