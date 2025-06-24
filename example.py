#!/usr/bin/env python3
"""
Quick Example Script for Astrophysical Signal Denoising

This script demonstrates the basic functionality of the denoising pipeline.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def main():
    """Run a quick example of signal generation and denoising."""
    
    print("="*60)
    print("ASTROPHYSICAL SIGNAL DENOISING - QUICK EXAMPLE")
    print("="*60)
    
    try:
        # Import our modules
        from signal_simulation import generate_gravitational_wave
        from noise_models import add_gaussian_noise
        
        print("✓ Modules imported successfully")
        
        # Generate a test signal
        print("\n1. Generating gravitational wave signal...")
        time, strain = generate_gravitational_wave(
            duration=1.0,
            sampling_rate=4096,
            m1=30.0,
            m2=25.0
        )
        print(f"   Signal length: {len(strain)} samples")
        print(f"   Duration: {time[-1]:.2f} seconds")
        print(f"   Peak strain: {np.max(np.abs(strain)):.2e}")
        
        # Add noise
        print("\n2. Adding noise...")
        noisy_signal = add_gaussian_noise(strain, snr=10)  # 10 dB SNR
        
        # Calculate actual SNR
        noise = noisy_signal - strain
        actual_snr = 10 * np.log10(np.mean(strain**2) / np.mean(noise**2))
        print(f"   Target SNR: 10.0 dB")
        print(f"   Actual SNR: {actual_snr:.2f} dB")
        
        # Create a simple plot
        print("\n3. Creating visualization...")
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(time, strain, 'b-', linewidth=1.5, label='Clean Signal')
        plt.plot(time, noisy_signal, 'r-', linewidth=0.8, alpha=0.7, label='Noisy Signal')
        plt.xlabel('Time (s)')
        plt.ylabel('Strain')
        plt.title('Gravitational Wave Signal')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Frequency domain
        from scipy import signal as sp_signal
        freqs, psd_clean = sp_signal.welch(strain, fs=4096)
        freqs, psd_noisy = sp_signal.welch(noisy_signal, fs=4096)
        
        plt.subplot(1, 2, 2)
        plt.loglog(freqs, psd_clean, 'b-', linewidth=1.5, label='Clean Signal')
        plt.loglog(freqs, psd_noisy, 'r-', linewidth=0.8, alpha=0.7, label='Noisy Signal')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power Spectral Density')
        plt.title('Frequency Domain')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('example_output.png', dpi=150, bbox_inches='tight')
        print(f"   Plot saved as 'example_output.png'")
        
        # Test model import
        print("\n4. Testing neural network models...")
        try:
            import torch
            from models import ConvAutoencoder
            
            model = ConvAutoencoder(input_length=4096)
            test_input = torch.randn(1, 1, 4096)
            output = model(test_input)
            
            print(f"   ✓ ConvAutoencoder model test successful")
            print(f"   Input shape: {test_input.shape}")
            print(f"   Output shape: {output.shape}")
            print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
            
        except ImportError as e:
            print(f"   ⚠ PyTorch not available: {e}")
            print("   Install PyTorch to enable neural network training")
        
        print("\n" + "="*60)
        print("EXAMPLE COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nNext steps:")
        print("1. Open 'notebooks/01_signal_denoising_demo.ipynb' for full tutorial")
        print("2. Run 'pip install -r requirements.txt' to install all dependencies")
        print("3. Customize parameters in 'config/config.yaml'")
        print("4. Generate larger datasets and train models")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure all dependencies are installed: pip install -r requirements.txt")
        print("2. Check Python path and module imports")
        print("3. Verify matplotlib backend is properly configured")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
