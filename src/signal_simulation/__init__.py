"""
Astrophysical Signal Simulation Module

This module provides functions to generate realistic astrophysical signals
including gravitational waves and pulsar signals.
"""

import numpy as np
from typing import Tuple, Optional, Union
import scipy.signal as signal
from astropy import units as u
from astropy.time import Time


def generate_gravitational_wave(
    duration: float = 1.0,
    sampling_rate: int = 4096,
    m1: float = 30.0,
    m2: float = 30.0,
    distance: float = 400.0,
    f_low: float = 20.0,
    approximant: str = "TaylorF2"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a simulated gravitational wave from a binary black hole merger.
    
    Parameters:
    -----------
    duration : float
        Duration of the signal in seconds
    sampling_rate : int
        Sampling rate in Hz
    m1, m2 : float
        Masses of the two objects in solar masses
    distance : float
        Luminosity distance in Mpc
    f_low : float
        Lower frequency cutoff in Hz
    approximant : str
        Waveform approximant to use
    
    Returns:
    --------
    time : np.ndarray
        Time array
    strain : np.ndarray
        Gravitational wave strain
    """
    # Time array
    time = np.linspace(0, duration, int(duration * sampling_rate))
    
    # Simplified gravitational wave model (chirp)
    # This is a basic implementation - for real applications, use LALSuite
    total_mass = m1 + m2
    chirp_mass = (m1 * m2)**(3/5) / total_mass**(1/5)
    
    # Convert to SI units
    G = 6.67430e-11  # m^3 kg^-1 s^-2
    c = 299792458    # m/s
    M_sun = 1.989e30 # kg
    
    chirp_mass_si = chirp_mass * M_sun
    distance_si = distance * 3.086e22  # Mpc to meters
    
    # Frequency evolution (simplified)
    f_start = f_low
    f_end = 2000.0  # Hz, upper frequency
    
    # Chirp time parameter
    tau = 5 * c**5 / (256 * np.pi**(8/3) * G**(5/3))
    tau *= (chirp_mass_si)**(-5/3) * f_start**(-8/3)
    
    # Frequency as function of time
    t_c = tau  # Coalescence time
    f_t = f_start * (1 - time/t_c)**(-3/8)
    f_t[f_t > f_end] = f_end
    
    # Phase evolution
    phase = 2 * np.pi * np.cumsum(f_t) / sampling_rate
    
    # Amplitude (simplified)
    amplitude = (G * chirp_mass_si)**(5/3) * (np.pi * f_t)**(2/3) / (c**4 * distance_si)
    amplitude *= 2  # Factor of 2 for optimal orientation
    
    # Generate strain
    strain = amplitude * np.cos(phase)
      # Apply window to avoid edge effects
    try:
        # Try scipy.signal.windows.tukey first (newer versions)
        from scipy.signal.windows import tukey
        window = tukey(len(strain), alpha=0.1)
    except ImportError:
        try:
            # Fallback to scipy.signal.tukey (older versions)
            window = signal.tukey(len(strain), alpha=0.1)
        except AttributeError:
            # Create a simple tapered window as fallback
            alpha = 0.1
            N = len(strain) 
            n_taper = int(alpha * N / 2)
            window = np.ones(N)
            # Taper the beginning
            window[:n_taper] = 0.5 * (1 + np.cos(np.pi * (np.arange(n_taper) / n_taper - 1)))
            # Taper the end
            window[-n_taper:] = 0.5 * (1 + np.cos(np.pi * np.arange(n_taper) / n_taper))
    
    strain *= window
    
    return time, strain


def generate_pulsar_signal(
    duration: float = 10.0,
    sampling_rate: int = 1024,
    period: float = 0.033,
    duty_cycle: float = 0.05,
    dm: float = 50.0,
    freq_center: float = 1400.0,
    bandwidth: float = 400.0,
    scintillation: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a simulated pulsar signal with dispersion and scintillation.
    
    Parameters:
    -----------
    duration : float
        Duration of the signal in seconds
    sampling_rate : int
        Sampling rate in Hz
    period : float
        Pulsar period in seconds
    duty_cycle : float
        Fraction of period where pulse is visible
    dm : float
        Dispersion measure in pc cm^-3
    freq_center : float
        Center frequency in MHz
    bandwidth : float
        Bandwidth in MHz
    scintillation : bool
        Whether to include scintillation effects
    
    Returns:
    --------
    time : np.ndarray
        Time array
    intensity : np.ndarray
        Pulsar signal intensity
    """
    # Time array
    time = np.linspace(0, duration, int(duration * sampling_rate))
    dt = 1.0 / sampling_rate
    
    # Generate pulse train
    pulse_times = np.arange(0, duration, period)
    intensity = np.zeros_like(time)
    
    # Pulse profile (Gaussian)
    pulse_width = period * duty_cycle
    sigma = pulse_width / 4  # 4-sigma width
    
    for pulse_time in pulse_times:
        # Add dispersion delay
        # Dispersion delay: t = 4.149 * DM * (1/f^2 - 1/f_ref^2) ms
        freq_low = freq_center - bandwidth/2
        dispersion_delay = 4.149e-3 * dm * (1/(freq_low**2) - 1/(freq_center**2))
        
        pulse_time_dispersed = pulse_time + dispersion_delay
        
        if pulse_time_dispersed >= 0 and pulse_time_dispersed < duration:
            # Gaussian pulse
            pulse_profile = np.exp(-0.5 * ((time - pulse_time_dispersed) / sigma)**2)
            intensity += pulse_profile
    
    # Add scintillation (amplitude modulation)
    if scintillation:
        # Simple scintillation model
        scint_timescale = 60.0  # seconds
        scint_freq = 1.0 / scint_timescale
        scint_modulation = 1 + 0.3 * np.sin(2 * np.pi * scint_freq * time)
        scint_modulation += 0.1 * np.sin(2 * np.pi * 3 * scint_freq * time)
        intensity *= scint_modulation
    
    # Add timing noise (random walk in phase)
    timing_noise_strength = 1e-8  # fractional
    timing_noise = np.cumsum(np.random.normal(0, timing_noise_strength, len(time)))
    
    # Apply timing noise as phase modulation
    phase_noise = 2 * np.pi * timing_noise / period
    intensity_shifted = np.interp(time, time + timing_noise * dt, intensity)
    
    return time, intensity_shifted


def add_astrophysical_background(
    signal: np.ndarray,
    background_type: str = "confusion_noise",
    strength: float = 0.1
) -> np.ndarray:
    """
    Add astrophysical background to a signal.
    
    Parameters:
    -----------
    signal : np.ndarray
        Input signal
    background_type : str
        Type of background ('confusion_noise', 'galactic_foreground')
    strength : float
        Relative strength of background
    
    Returns:
    --------
    np.ndarray
        Signal with astrophysical background
    """
    if background_type == "confusion_noise":
        # Galactic binary confusion noise (simplified)
        # Power law spectrum typical of galactic binaries
        freq = np.fft.fftfreq(len(signal))
        freq[0] = 1e-10  # Avoid division by zero
        
        # Power spectral density ~ f^(-7/3)
        psd = np.abs(freq)**(-7/3)
        psd[0] = psd[1]  # Fix DC component
        
        # Generate colored noise
        white_noise = np.random.normal(0, 1, len(signal))
        noise_fft = np.fft.fft(white_noise)
        colored_noise_fft = noise_fft * np.sqrt(psd)
        colored_noise = np.real(np.fft.ifft(colored_noise_fft))
        
        # Normalize and scale
        colored_noise = colored_noise / np.std(colored_noise)
        background = strength * np.std(signal) * colored_noise
        
    elif background_type == "galactic_foreground":
        # Simplified galactic foreground
        # Multiple sinusoidal components
        time = np.arange(len(signal))
        background = np.zeros_like(signal)
        
        for i in range(10):
            freq = np.random.uniform(0.1, 10) / len(signal)
            amp = strength * np.std(signal) * np.random.exponential(0.1)
            phase = np.random.uniform(0, 2*np.pi)
            background += amp * np.sin(2 * np.pi * freq * time + phase)
    
    else:
        raise ValueError(f"Unknown background type: {background_type}")
    
    return signal + background


def generate_realistic_detector_response(
    signal: np.ndarray,
    detector: str = "LIGO",
    antenna_pattern: Tuple[float, float] = (0.7, 0.3)
) -> np.ndarray:
    """
    Apply realistic detector response to a gravitational wave signal.
    
    Parameters:
    -----------
    signal : np.ndarray
        Input gravitational wave signal
    detector : str
        Detector name ('LIGO', 'Virgo', 'KAGRA')
    antenna_pattern : tuple
        (F_plus, F_cross) antenna pattern functions
    
    Returns:
    --------
    np.ndarray
        Signal with detector response applied
    """
    F_plus, F_cross = antenna_pattern
    
    # For simplicity, assume the signal is h_plus polarization
    # In reality, you'd need both polarizations and proper detector response
    detector_signal = F_plus * signal
    
    # Add detector-specific characteristics
    if detector == "LIGO":
        # Apply transfer function (simplified)
        # Real LIGO has complex transfer function
        pass
    elif detector == "Virgo":
        # Virgo-specific response
        pass
    elif detector == "KAGRA":
        # KAGRA-specific response
        pass
    
    return detector_signal


if __name__ == "__main__":
    # Example usage
    import matplotlib.pyplot as plt
    
    # Generate gravitational wave
    print("Generating gravitational wave...")
    time_gw, strain = generate_gravitational_wave(duration=1.0)
    
    # Generate pulsar signal
    print("Generating pulsar signal...")
    time_psr, intensity = generate_pulsar_signal(duration=5.0)
    
    # Plot examples
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    ax1.plot(time_gw, strain)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Strain')
    ax1.set_title('Gravitational Wave Signal')
    ax1.grid(True)
    
    ax2.plot(time_psr, intensity)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Intensity')
    ax2.set_title('Pulsar Signal')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('example_signals.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Example signals generated and saved as 'example_signals.png'")
