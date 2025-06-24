# Astrophysical Signal Denoising Project

## 🚀 Project Overview

This project demonstrates training neural networks (autoencoders) to denoise simulated astrophysical signals such as **gravitational waves** and **pulsar signals**. The implementation showcases the network's ability to reconstruct original signals from noisy data and provides comprehensive performance quantification.

## 📁 Project Structure

```
The_clarifyer/
├── src/                          # Source code modules
│   ├── signal_simulation/        # Signal generation (GW, pulsars)
│   ├── noise_models/            # Noise addition and modeling
│   ├── models/                  # Neural network architectures
│   ├── training/                # Training pipelines
│   ├── evaluation/              # Performance metrics
│   └── utils/                   # Utility functions
├── notebooks/                   # Jupyter notebooks
│   └── 01_signal_denoising_demo.ipynb
├── config/                      # Configuration files
│   └── config.yaml
├── data/                        # Generated datasets (created during use)
├── models/                      # Saved model checkpoints (created during training)
├── results/                     # Training results and plots (created during use)
├── requirements.txt             # Python dependencies
├── example.py                   # Quick start example
└── README.md                    # This file
```

## 🔬 Key Features

### Signal Types
- **Gravitational Waves**: Binary black hole mergers, neutron star mergers
- **Pulsar Signals**: Periodic pulses with realistic timing noise and dispersion

### Neural Network Architectures
- **1D Convolutional Autoencoders**: Standard deep learning approach
- **U-Net Autoencoders**: Skip connections for better reconstruction
- **LSTM Autoencoders**: Recurrent networks for temporal dependencies
- **WaveNet-inspired**: Dilated convolutions for multi-scale features

### Noise Models
- **Gaussian Noise**: White noise with controlled SNR
- **Colored Noise**: Realistic detector noise (LIGO PSD models)
- **Instrumental Glitches**: Transient artifacts (blips, whistles)
- **Line Noise**: Power line harmonics

### Performance Metrics
- **Signal-to-Noise Ratio (SNR)** improvement
- **Mean Squared Error (MSE)** and variations
- **Structural Similarity Index (SSIM)**
- **Cross-correlation** coefficients
- **Spectral fidelity** measures

## 🛠️ Setup Instructions

### Prerequisites
- Python 3.8 or higher
- Visual Studio Code (recommended)
- Git (for version control)

### Installation

1. **Clone or download this project** to your local machine

2. **Open in VS Code**:
   ```bash
   code The_clarifyer
   ```

3. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation** by running the example:
   ```bash
   python example.py
   ```

### Required Python Packages
- **Deep Learning**: PyTorch, TensorFlow
- **Scientific Computing**: NumPy, SciPy, Matplotlib
- **Astrophysics**: Astropy, GWPy (optional)
- **Notebooks**: Jupyter, IPython
- **Utilities**: tqdm, PyYAML, h5py

## 🚀 Quick Start

### 1. Run the Example Script
```bash
python example.py
```
This generates a sample gravitational wave, adds noise, and creates visualizations.

### 2. Open the Demo Notebook
Launch Jupyter and open `notebooks/01_signal_denoising_demo.ipynb` for a comprehensive tutorial that covers:
- Signal generation and visualization
- Noise modeling and application
- Neural network training
- Performance evaluation
- Results analysis

### 3. Customize Configuration
Edit `config/config.yaml` to adjust:
- Model architectures and hyperparameters
- Training settings (epochs, batch size, learning rate)
- Signal generation parameters
- Noise characteristics

## 🔍 Usage Examples

### Generate Signals
```python
from src.signal_simulation import generate_gravitational_wave, generate_pulsar_signal

# Gravitational wave from binary black hole merger
time, strain = generate_gravitational_wave(
    duration=1.0,
    sampling_rate=4096,
    m1=30.0,  # Solar masses
    m2=25.0,  # Solar masses
    distance=400.0  # Mpc
)

# Pulsar signal
time, intensity = generate_pulsar_signal(
    duration=5.0,
    period=0.033,  # 33 ms (Crab-like)
    duty_cycle=0.05
)
```

### Add Realistic Noise
```python
from src.noise_models import add_gaussian_noise, add_colored_noise

# Add Gaussian white noise
noisy_signal = add_gaussian_noise(clean_signal, snr=10)  # 10 dB SNR

# Add realistic detector noise
noisy_signal = add_colored_noise(
    clean_signal, 
    snr=10, 
    psd_model="LIGO_design"
)
```

### Train Denoising Models
```python
from src.models import ConvAutoencoder
from src.training import ModelTrainer

# Initialize model
model = ConvAutoencoder(input_length=4096)

# Train
trainer = ModelTrainer(model)
history = trainer.train(
    train_dataset=train_data,
    val_dataset=val_data,
    epochs=100
)
```

### Evaluate Performance
```python
from src.evaluation import SignalEvaluator

evaluator = SignalEvaluator(sampling_rate=4096)
metrics = evaluator.comprehensive_evaluation(
    original_signal,
    noisy_signal, 
    denoised_signal
)

print(f"SNR improvement: {metrics['snr_metrics']['snr_improvement_db']:.2f} dB")
```

## 📊 Expected Results

The trained models typically achieve:
- **SNR improvements** of 5-15 dB depending on input noise level
- **Correlation coefficients** > 0.8 with original signals
- **Successful removal** of instrumental glitches and line noise
- **Preservation** of astrophysical signal characteristics

## 🔬 Scientific Applications

This technology can be applied to:
- **Gravitational Wave Detection**: Improve sensitivity of LIGO/Virgo/KAGRA
- **Pulsar Timing Arrays**: Clean timing data for gravitational wave searches
- **Transient Detection**: Identify buried astrophysical signals
- **Real-time Processing**: Online denoising for live detector data

## 🎯 Advanced Features

### Model Architectures
- **Skip connections** (U-Net) for better gradient flow
- **Attention mechanisms** for focusing on signal features
- **Multi-scale processing** with dilated convolutions
- **Uncertainty quantification** for reliability assessment

### Training Enhancements
- **Data augmentation** with varied noise types
- **Transfer learning** between signal types
- **Adversarial training** for robustness
- **Ensemble methods** for improved performance

### Performance Analysis
- **SNR sensitivity curves** across noise levels
- **Spectral analysis** of reconstruction fidelity
- **Statistical significance** testing
- **Computational efficiency** benchmarks

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional signal types (neutron star mergers, supernovae)
- More sophisticated noise models
- Real detector data integration
- Performance optimizations
- Documentation enhancements

## 📚 References

- **LIGO Scientific Collaboration**: Gravitational wave detection methods
- **Pulsar Search Collaboratory**: Pulsar signal processing techniques
- **Deep Learning Literature**: Autoencoder architectures for signal processing
- **Astrophysical Data Analysis**: Noise characterization and signal extraction

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔗 Related Resources

- [LIGO Open Science Center](https://www.gw-openscience.org/)
- [Pulsar Search Collaboratory](https://pulsar.nao.ac.jp/psc/)
- [GWPy Documentation](https://gwpy.github.io/)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)

## ⚠️ Important Notes

- **Computational Requirements**: Training neural networks requires significant computational resources
- **Data Storage**: Generated datasets can be large (GB-scale for comprehensive training)
- **Real Data**: This project uses simulated signals; real detector data has additional complexities
- **Scientific Validation**: Results should be validated against known astrophysical standards

---

## 🎉 Get Started Now!

1. **Install dependencies**: `pip install -r requirements.txt`
2. **Run the example**: `python example.py`
3. **Open the demo notebook**: `notebooks/01_signal_denoising_demo.ipynb`
4. **Start experimenting** with your own signals and models!

Happy denoising! 🌟
