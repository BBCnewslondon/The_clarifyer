<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->

# Astrophysical Signal Denoising Project

This project focuses on training neural networks (autoencoders) to denoise simulated astrophysical signals such as gravitational waves and pulsar signals. The codebase includes:

## Key Components:
- Signal simulation modules for gravitational waves and pulsar signals
- Noise generation and signal contamination utilities
- Autoencoder neural network architectures for denoising
- Training pipelines with data preprocessing
- Performance evaluation metrics and visualization tools
- Jupyter notebooks for experimentation and demonstration

## Coding Guidelines:
- Use PyTorch or TensorFlow for neural network implementation
- Follow scientific computing best practices with NumPy and SciPy
- Include comprehensive documentation for signal processing functions
- Implement robust error handling for data processing pipelines
- Use type hints for better code maintainability
- Include unit tests for critical signal processing functions

## Signal Processing Focus:
- Gravitational wave signals: chirp masses, strain data, frequency evolution
- Pulsar signals: periodic signals, dispersion, timing noise
- Noise models: Gaussian noise, colored noise, realistic detector noise
- Performance metrics: SNR improvement, signal reconstruction fidelity, MSE/SSIM
