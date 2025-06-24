"""
Evaluation Metrics and Visualization for Signal Denoising

This module provides comprehensive evaluation tools for assessing
the performance of denoising models on astrophysical signals.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal, stats
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
from typing import Tuple, Dict, List, Optional, Union
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd


class SignalEvaluator:
    """
    Comprehensive evaluation class for signal denoising performance.
    """
    
    def __init__(self, sampling_rate: int = 4096):
        """
        Initialize the evaluator.
        
        Parameters:
        -----------
        sampling_rate : int
            Sampling rate of signals in Hz
        """
        self.sampling_rate = sampling_rate
        
    def calculate_snr(
        self,
        signal: np.ndarray,
        noise: np.ndarray
    ) -> float:
        """
        Calculate Signal-to-Noise Ratio.
        
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
    
    def calculate_snr_improvement(
        self,
        original: np.ndarray,
        noisy: np.ndarray,
        denoised: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate SNR improvement from denoising.
        
        Parameters:
        -----------
        original : np.ndarray
            Original clean signal
        noisy : np.ndarray
            Noisy signal
        denoised : np.ndarray
            Denoised signal
        
        Returns:
        --------
        dict
            SNR metrics including improvement
        """
        # Calculate noise components
        input_noise = noisy - original
        output_noise = denoised - original
        
        # Calculate SNRs
        input_snr = self.calculate_snr(original, input_noise)
        output_snr = self.calculate_snr(original, output_noise)
        snr_improvement = output_snr - input_snr
        
        return {
            'input_snr_db': input_snr,
            'output_snr_db': output_snr,
            'snr_improvement_db': snr_improvement
        }
    
    def calculate_correlation_metrics(
        self,
        original: np.ndarray,
        denoised: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate correlation-based metrics.
        
        Parameters:
        -----------
        original : np.ndarray
            Original clean signal
        denoised : np.ndarray
            Denoised signal
        
        Returns:
        --------
        dict
            Correlation metrics
        """
        # Pearson correlation
        pearson_corr, pearson_p = stats.pearsonr(original, denoised)
        
        # Cross-correlation
        cross_corr = signal.correlate(original, denoised, mode='full')
        max_cross_corr = np.max(cross_corr) / (np.linalg.norm(original) * np.linalg.norm(denoised))
        
        # Normalized cross-correlation
        normalized_cross_corr = np.max(signal.correlate(
            original / np.linalg.norm(original),
            denoised / np.linalg.norm(denoised),
            mode='full'
        ))
        
        return {
            'pearson_correlation': pearson_corr,
            'pearson_p_value': pearson_p,
            'max_cross_correlation': max_cross_corr,
            'normalized_cross_correlation': normalized_cross_corr
        }
    
    def calculate_reconstruction_metrics(
        self,
        original: np.ndarray,
        denoised: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate reconstruction quality metrics.
        
        Parameters:
        -----------
        original : np.ndarray
            Original clean signal
        denoised : np.ndarray
            Denoised signal
        
        Returns:
        --------
        dict
            Reconstruction metrics
        """
        # Mean Squared Error
        mse = mean_squared_error(original, denoised)
        
        # Root Mean Squared Error
        rmse = np.sqrt(mse)
        
        # Mean Absolute Error
        mae = mean_absolute_error(original, denoised)
        
        # Normalized RMSE
        signal_range = np.max(original) - np.min(original)
        nrmse = rmse / signal_range if signal_range > 0 else np.inf
        
        # Peak Signal-to-Noise Ratio
        max_signal = np.max(original**2)
        psnr = 10 * np.log10(max_signal / mse) if mse > 0 else np.inf
        
        # Structural Similarity Index (simplified 1D version)
        ssim = self._calculate_ssim_1d(original, denoised)
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'nrmse': nrmse,
            'psnr_db': psnr,
            'ssim': ssim
        }
    
    def _calculate_ssim_1d(
        self,
        x: np.ndarray,
        y: np.ndarray,
        window_size: int = 11,
        sigma: float = 1.5
    ) -> float:
        """
        Calculate 1D Structural Similarity Index.
        
        Parameters:
        -----------
        x, y : np.ndarray
            Input signals
        window_size : int
            Size of sliding window
        sigma : float
            Standard deviation for Gaussian window
        
        Returns:
        --------
        float
            SSIM value between -1 and 1
        """
        # Constants for stability
        C1 = 0.01**2
        C2 = 0.03**2
          # Create Gaussian window (using scipy.signal.windows.gaussian for compatibility)
        try:
            from scipy.signal.windows import gaussian
            window = gaussian(window_size, sigma)
        except ImportError:
            # Fallback for older scipy versions
            try:
                window = signal.gaussian(window_size, sigma)
            except AttributeError:
                # Manual Gaussian window implementation
                n = np.arange(window_size)
                window = np.exp(-0.5 * ((n - (window_size - 1) / 2) / sigma) ** 2)
        window = window / np.sum(window)
        
        # Apply window convolution
        mu1 = signal.convolve(x, window, mode='valid')
        mu2 = signal.convolve(y, window, mode='valid')
        
        mu1_sq = mu1**2
        mu2_sq = mu2**2
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = signal.convolve(x**2, window, mode='valid') - mu1_sq
        sigma2_sq = signal.convolve(y**2, window, mode='valid') - mu2_sq
        sigma12 = signal.convolve(x * y, window, mode='valid') - mu1_mu2
        
        # SSIM calculation
        numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
        denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        
        ssim_map = numerator / denominator
        return np.mean(ssim_map)
    
    def calculate_spectral_metrics(
        self,
        original: np.ndarray,
        denoised: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate frequency domain metrics.
        
        Parameters:
        -----------
        original : np.ndarray
            Original clean signal
        denoised : np.ndarray
            Denoised signal
        
        Returns:
        --------
        dict
            Spectral metrics
        """
        # Compute power spectral densities
        freqs_orig, psd_orig = signal.welch(original, fs=self.sampling_rate, nperseg=len(original)//4)
        freqs_denoised, psd_denoised = signal.welch(denoised, fs=self.sampling_rate, nperseg=len(denoised)//4)
        
        # Spectral correlation
        spectral_correlation = np.corrcoef(psd_orig, psd_denoised)[0, 1]
        
        # Spectral distortion (log-spectral distance)
        log_spectral_distance = np.mean((np.log10(psd_orig + 1e-10) - np.log10(psd_denoised + 1e-10))**2)
        
        # Spectral centroid preservation
        spectral_centroid_orig = np.sum(freqs_orig * psd_orig) / np.sum(psd_orig)
        spectral_centroid_denoised = np.sum(freqs_denoised * psd_denoised) / np.sum(psd_denoised)
        centroid_difference = abs(spectral_centroid_orig - spectral_centroid_denoised)
        
        # Bandwidth preservation
        spectral_spread_orig = np.sqrt(np.sum(((freqs_orig - spectral_centroid_orig)**2) * psd_orig) / np.sum(psd_orig))
        spectral_spread_denoised = np.sqrt(np.sum(((freqs_denoised - spectral_centroid_denoised)**2) * psd_denoised) / np.sum(psd_denoised))
        spread_difference = abs(spectral_spread_orig - spectral_spread_denoised)
        
        return {
            'spectral_correlation': spectral_correlation,
            'log_spectral_distance': log_spectral_distance,
            'centroid_difference_hz': centroid_difference,
            'spread_difference_hz': spread_difference
        }
    
    def comprehensive_evaluation(
        self,
        original: np.ndarray,
        noisy: np.ndarray,
        denoised: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """
        Perform comprehensive evaluation of denoising performance.
        
        Parameters:
        -----------
        original : np.ndarray
            Original clean signal
        noisy : np.ndarray
            Noisy signal
        denoised : np.ndarray
            Denoised signal
        
        Returns:
        --------
        dict
            Comprehensive evaluation metrics
        """
        results = {
            'snr_metrics': self.calculate_snr_improvement(original, noisy, denoised),
            'correlation_metrics': self.calculate_correlation_metrics(original, denoised),
            'reconstruction_metrics': self.calculate_reconstruction_metrics(original, denoised),
            'spectral_metrics': self.calculate_spectral_metrics(original, denoised)
        }
        
        return results
    
    def plot_reconstruction_comparison(
        self,
        original: np.ndarray,
        noisy: np.ndarray,
        denoised: np.ndarray,
        time_range: Optional[Tuple[float, float]] = None,
        title: str = "Signal Reconstruction Comparison"
    ) -> plt.Figure:
        """
        Plot comparison of original, noisy, and denoised signals.
        
        Parameters:
        -----------
        original : np.ndarray
            Original clean signal
        noisy : np.ndarray
            Noisy signal
        denoised : np.ndarray
            Denoised signal
        time_range : tuple, optional
            Time range to plot (start, end) in seconds
        title : str
            Plot title
        
        Returns:
        --------
        plt.Figure
            Matplotlib figure
        """
        time = np.arange(len(original)) / self.sampling_rate
        
        if time_range:
            start_idx = int(time_range[0] * self.sampling_rate)
            end_idx = int(time_range[1] * self.sampling_rate)
            time = time[start_idx:end_idx]
            original = original[start_idx:end_idx]
            noisy = noisy[start_idx:end_idx]
            denoised = denoised[start_idx:end_idx]
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # Original signal
        axes[0].plot(time, original, 'b-', linewidth=1, label='Original')
        axes[0].set_ylabel('Amplitude')
        axes[0].set_title(f'{title} - Original Signal')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        # Noisy signal
        axes[1].plot(time, noisy, 'r-', linewidth=0.8, alpha=0.7, label='Noisy')
        axes[1].plot(time, original, 'b-', linewidth=1, alpha=0.8, label='Original')
        axes[1].set_ylabel('Amplitude')
        axes[1].set_title(f'{title} - Noisy Signal')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        
        # Denoised signal
        axes[2].plot(time, denoised, 'g-', linewidth=1, label='Denoised')
        axes[2].plot(time, original, 'b-', linewidth=1, alpha=0.8, label='Original')
        axes[2].set_xlabel('Time (s)')
        axes[2].set_ylabel('Amplitude')
        axes[2].set_title(f'{title} - Denoised Signal')
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()
        
        plt.tight_layout()
        return fig
    
    def plot_spectral_comparison(
        self,
        original: np.ndarray,
        noisy: np.ndarray,
        denoised: np.ndarray,
        title: str = "Spectral Comparison"
    ) -> plt.Figure:
        """
        Plot power spectral density comparison.
        
        Parameters:
        -----------
        original : np.ndarray
            Original clean signal
        noisy : np.ndarray
            Noisy signal
        denoised : np.ndarray
            Denoised signal
        title : str
            Plot title
        
        Returns:
        --------
        plt.Figure
            Matplotlib figure
        """
        # Compute PSDs
        freqs_orig, psd_orig = signal.welch(original, fs=self.sampling_rate)
        freqs_noisy, psd_noisy = signal.welch(noisy, fs=self.sampling_rate)
        freqs_denoised, psd_denoised = signal.welch(denoised, fs=self.sampling_rate)
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Linear scale
        axes[0].plot(freqs_orig, psd_orig, 'b-', label='Original', linewidth=2)
        axes[0].plot(freqs_noisy, psd_noisy, 'r-', alpha=0.7, label='Noisy', linewidth=1)
        axes[0].plot(freqs_denoised, psd_denoised, 'g-', label='Denoised', linewidth=2)
        axes[0].set_xlabel('Frequency (Hz)')
        axes[0].set_ylabel('Power Spectral Density')
        axes[0].set_title(f'{title} - Linear Scale')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Log scale
        axes[1].loglog(freqs_orig, psd_orig, 'b-', label='Original', linewidth=2)
        axes[1].loglog(freqs_noisy, psd_noisy, 'r-', alpha=0.7, label='Noisy', linewidth=1)
        axes[1].loglog(freqs_denoised, psd_denoised, 'g-', label='Denoised', linewidth=2)
        axes[1].set_xlabel('Frequency (Hz)')
        axes[1].set_ylabel('Power Spectral Density')
        axes[1].set_title(f'{title} - Log Scale')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_metrics_summary(
        self,
        metrics: Dict[str, Dict[str, float]],
        title: str = "Performance Metrics Summary"
    ) -> plt.Figure:
        """
        Plot summary of evaluation metrics.
        
        Parameters:
        -----------
        metrics : dict
            Comprehensive evaluation metrics
        title : str
            Plot title
        
        Returns:
        --------
        plt.Figure
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # SNR Metrics
        snr_data = metrics['snr_metrics']
        snr_labels = ['Input SNR', 'Output SNR', 'Improvement']
        snr_values = [snr_data['input_snr_db'], snr_data['output_snr_db'], snr_data['snr_improvement_db']]
        
        bars = axes[0, 0].bar(snr_labels, snr_values, color=['red', 'green', 'blue'])
        axes[0, 0].set_ylabel('SNR (dB)')
        axes[0, 0].set_title('SNR Metrics')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, snr_values):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                           f'{value:.2f}', ha='center', va='bottom')
        
        # Correlation Metrics
        corr_data = metrics['correlation_metrics']
        corr_labels = ['Pearson', 'Max Cross-Corr', 'Norm Cross-Corr']
        corr_values = [corr_data['pearson_correlation'], 
                      corr_data['max_cross_correlation'],
                      corr_data['normalized_cross_correlation']]
        
        bars = axes[0, 1].bar(corr_labels, corr_values, color=['purple', 'orange', 'cyan'])
        axes[0, 1].set_ylabel('Correlation')
        axes[0, 1].set_title('Correlation Metrics')
        axes[0, 1].set_ylim([0, 1])
        axes[0, 1].grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, corr_values):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
        
        # Reconstruction Metrics
        recon_data = metrics['reconstruction_metrics']
        recon_labels = ['MSE', 'RMSE', 'MAE', 'NRMSE']
        recon_values = [recon_data['mse'], recon_data['rmse'], 
                       recon_data['mae'], recon_data['nrmse']]
        
        bars = axes[1, 0].bar(recon_labels, recon_values, color=['brown', 'pink', 'gray', 'olive'])
        axes[1, 0].set_ylabel('Error')
        axes[1, 0].set_title('Reconstruction Metrics')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Spectral Metrics
        spectral_data = metrics['spectral_metrics']
        spectral_labels = ['Spectral Corr', 'Log Spectral Dist']
        spectral_values = [spectral_data['spectral_correlation'],
                          spectral_data['log_spectral_distance']]
        
        bars = axes[1, 1].bar(spectral_labels, spectral_values, color=['teal', 'gold'])
        axes[1, 1].set_ylabel('Value')
        axes[1, 1].set_title('Spectral Metrics')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, spectral_values):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        return fig


def evaluate_model_on_dataset(
    model: torch.nn.Module,
    test_dataset: torch.utils.data.Dataset,
    device: str = 'cpu',
    batch_size: int = 32,
    sampling_rate: int = 4096
) -> Dict[str, List[float]]:
    """
    Evaluate a trained model on a test dataset.
    
    Parameters:
    -----------
    model : torch.nn.Module
        Trained denoising model
    test_dataset : Dataset
        Test dataset
    device : str
        Device to run evaluation on
    batch_size : int
        Batch size for evaluation
    sampling_rate : int
        Sampling rate of signals
    
    Returns:
    --------
    dict
        Aggregated evaluation metrics
    """
    model.eval()
    evaluator = SignalEvaluator(sampling_rate=sampling_rate)
    
    # Storage for metrics
    all_metrics = {
        'snr_improvement': [],
        'pearson_correlation': [],
        'mse': [],
        'ssim': [],
        'spectral_correlation': []
    }
    
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    with torch.no_grad():
        for batch_idx, (noisy_batch, clean_batch) in enumerate(test_loader):
            noisy_batch = noisy_batch.to(device)
            clean_batch = clean_batch.to(device)
            
            # Get model predictions
            denoised_batch = model(noisy_batch)
            
            # Convert to numpy for evaluation
            for i in range(noisy_batch.size(0)):
                original = clean_batch[i, 0].cpu().numpy()
                noisy = noisy_batch[i, 0].cpu().numpy()
                denoised = denoised_batch[i, 0].cpu().numpy()
                
                # Calculate metrics
                metrics = evaluator.comprehensive_evaluation(original, noisy, denoised)
                
                # Store key metrics
                all_metrics['snr_improvement'].append(metrics['snr_metrics']['snr_improvement_db'])
                all_metrics['pearson_correlation'].append(metrics['correlation_metrics']['pearson_correlation'])
                all_metrics['mse'].append(metrics['reconstruction_metrics']['mse'])
                all_metrics['ssim'].append(metrics['reconstruction_metrics']['ssim'])
                all_metrics['spectral_correlation'].append(metrics['spectral_metrics']['spectral_correlation'])
    
    return all_metrics


if __name__ == "__main__":
    # Example usage
    print("Evaluation module - run as part of full pipeline")
