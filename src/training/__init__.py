"""
Training Pipeline for Signal Denoising Models

This module provides training utilities and pipelines for neural network models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List, Optional, Callable
import os
import pickle
from tqdm import tqdm
import yaml
from datetime import datetime


class SignalDataset(Dataset):
    """
    Dataset class for astrophysical signals.
    """
    
    def __init__(
        self,
        clean_signals: np.ndarray,
        noisy_signals: np.ndarray,
        transform: Optional[Callable] = None
    ):
        """
        Initialize the signal dataset.
        
        Parameters:
        -----------
        clean_signals : np.ndarray
            Array of clean signals
        noisy_signals : np.ndarray
            Array of noisy signals
        transform : callable, optional
            Optional transform to be applied
        """
        assert len(clean_signals) == len(noisy_signals), "Mismatch in signal counts"
        
        self.clean_signals = clean_signals
        self.noisy_signals = noisy_signals
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.clean_signals)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        clean = self.clean_signals[idx]
        noisy = self.noisy_signals[idx]
        
        if self.transform:
            clean = self.transform(clean)
            noisy = self.transform(noisy)
        
        # Convert to tensors and add channel dimension
        clean_tensor = torch.FloatTensor(clean).unsqueeze(0)  # Shape: (1, length)
        noisy_tensor = torch.FloatTensor(noisy).unsqueeze(0)
        
        return noisy_tensor, clean_tensor


class ModelTrainer:
    """
    Training pipeline for denoising models.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'auto',
        save_dir: str = 'models'
    ):
        """
        Initialize the model trainer.
        
        Parameters:
        -----------
        model : nn.Module
            The neural network model to train
        device : str
            Device to use ('cuda', 'cpu', or 'auto')
        save_dir : str
            Directory to save model checkpoints
        """
        self.model = model
        self.device = self._get_device(device)
        self.save_dir = save_dir
        self.model.to(self.device)
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
    def _get_device(self, device: str) -> str:
        """Get the appropriate device."""
        if device == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return device
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        epoch: int
    ) -> float:
        """
        Train for one epoch.
        
        Parameters:
        -----------
        train_loader : DataLoader
            Training data loader
        optimizer : optim.Optimizer
            Optimizer for training
        criterion : nn.Module
            Loss function
        epoch : int
            Current epoch number
        
        Returns:
        --------
        float
            Average training loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = len(train_loader)
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}', leave=False)
        
        for batch_idx, (noisy, clean) in enumerate(progress_bar):
            noisy = noisy.to(self.device)
            clean = clean.to(self.device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            output = self.model(noisy)
            loss = criterion(output, clean)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            avg_loss = total_loss / (batch_idx + 1)
            
            # Update progress bar
            progress_bar.set_postfix({'Loss': f'{avg_loss:.6f}'})
        
        return total_loss / num_batches
    
    def validate_epoch(
        self,
        val_loader: DataLoader,
        criterion: nn.Module
    ) -> float:
        """
        Validate for one epoch.
        
        Parameters:
        -----------
        val_loader : DataLoader
            Validation data loader
        criterion : nn.Module
            Loss function
        
        Returns:
        --------
        float
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = len(val_loader)
        
        with torch.no_grad():
            for noisy, clean in val_loader:
                noisy = noisy.to(self.device)
                clean = clean.to(self.device)
                
                output = self.model(noisy)
                loss = criterion(output, clean)
                
                total_loss += loss.item()
        
        return total_loss / num_batches
    
    def train(
        self,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        optimizer_type: str = 'adam',
        criterion_type: str = 'mse',
        scheduler_type: Optional[str] = None,
        early_stopping_patience: int = 10,
        save_best: bool = True,
        plot_training: bool = True
    ) -> Dict[str, List[float]]:
        """
        Train the model.
        
        Parameters:
        -----------
        train_dataset : Dataset
            Training dataset
        val_dataset : Dataset, optional
            Validation dataset
        epochs : int
            Number of training epochs
        batch_size : int
            Batch size for training
        learning_rate : float
            Learning rate
        optimizer_type : str
            Optimizer type ('adam', 'sgd', 'rmsprop')
        criterion_type : str
            Loss function type ('mse', 'mae', 'huber')
        scheduler_type : str, optional
            Learning rate scheduler ('step', 'cosine', 'plateau')
        early_stopping_patience : int
            Early stopping patience
        save_best : bool
            Whether to save the best model
        plot_training : bool
            Whether to plot training curves
        
        Returns:
        --------
        dict
            Training history
        """
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) if val_dataset else None
        
        # Initialize optimizer
        optimizer = self._get_optimizer(optimizer_type, learning_rate)
        
        # Initialize criterion
        criterion = self._get_criterion(criterion_type)
        
        # Initialize scheduler
        scheduler = self._get_scheduler(scheduler_type, optimizer) if scheduler_type else None
        
        # Training loop
        patience_counter = 0
        
        print(f"Starting training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(epochs):
            # Train epoch
            train_loss = self.train_epoch(train_loader, optimizer, criterion, epoch + 1)
            self.train_losses.append(train_loss)
              # Validate epoch
            if val_loader:
                val_loss = self.validate_epoch(val_loader, criterion)
                self.val_losses.append(val_loss)
                
                print(f'Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
                
                # Early stopping and best model saving
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    patience_counter = 0
                    
                    if save_best:
                        self.save_model('best_model.pth', epoch + 1, train_loss, val_loss)
                else:
                    patience_counter += 1
                
                if patience_counter >= early_stopping_patience:
                    print(f'Early stopping at epoch {epoch+1}')
                    break
            else:
                print(f'Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}')
            
            # Update learning rate
            if scheduler:
                if scheduler_type == 'plateau' and val_loader:
                    scheduler.step(val_loss)
                else:
                    scheduler.step()
        
        # Save final model
        if save_best:
            final_val_loss = self.val_losses[-1] if val_loader else None
            self.save_model('final_model.pth', epochs, self.train_losses[-1], final_val_loss)
        
        # Plot training curves
        if plot_training:
            self.plot_training_curves()
        
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses if val_loader else None
        }
        
        return history
    
    def _get_optimizer(self, optimizer_type: str, learning_rate: float) -> optim.Optimizer:
        """Get optimizer."""
        if optimizer_type == 'adam':
            return optim.Adam(self.model.parameters(), lr=learning_rate)
        elif optimizer_type == 'sgd':
            return optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9)
        elif optimizer_type == 'rmsprop':
            return optim.RMSprop(self.model.parameters(), lr=learning_rate)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")
    
    def _get_criterion(self, criterion_type: str) -> nn.Module:
        """Get loss function."""
        if criterion_type == 'mse':
            return nn.MSELoss()
        elif criterion_type == 'mae':
            return nn.L1Loss()
        elif criterion_type == 'huber':
            return nn.SmoothL1Loss()
        else:
            raise ValueError(f"Unknown criterion: {criterion_type}")
    
    def _get_scheduler(self, scheduler_type: str, optimizer: optim.Optimizer):
        """Get learning rate scheduler."""
        if scheduler_type == 'step':
            return optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        elif scheduler_type == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
        elif scheduler_type == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_type}")
    
    def save_model(
        self,
        filename: str,
        epoch: int,
        train_loss: float,
        val_loss: Optional[float] = None
    ):
        """Save model checkpoint."""
        filepath = os.path.join(self.save_dir, filename)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'model_architecture': str(self.model)
        }
        
        torch.save(checkpoint, filepath)
        print(f'Model saved to {filepath}')
    
    def load_model(self, filepath: str) -> Dict:
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Restore training history
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        
        print(f'Model loaded from {filepath}')
        return checkpoint
    
    def plot_training_curves(self):
        """Plot training and validation curves."""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Training Loss')
        if self.val_losses:
            plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Progress')
        plt.legend()
        plt.grid(True)
        
        if len(self.train_losses) > 10:
            plt.subplot(1, 2, 2)
            plt.plot(self.train_losses[10:], label='Training Loss (from epoch 10)')
            if self.val_losses:
                plt.plot(self.val_losses[10:], label='Validation Loss (from epoch 10)')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training Progress (Zoomed)')
            plt.legend()
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'training_curves.png'), dpi=150, bbox_inches='tight')
        plt.show()


def create_datasets(
    clean_signals: np.ndarray,
    noisy_signals: np.ndarray,
    train_split: float = 0.8,
    val_split: float = 0.1,
    test_split: float = 0.1
) -> Tuple[SignalDataset, SignalDataset, SignalDataset]:
    """
    Create train, validation, and test datasets.
    
    Parameters:
    -----------
    clean_signals : np.ndarray
        Array of clean signals
    noisy_signals : np.ndarray
        Array of noisy signals
    train_split : float
        Fraction for training
    val_split : float
        Fraction for validation
    test_split : float
        Fraction for testing
    
    Returns:
    --------
    tuple
        (train_dataset, val_dataset, test_dataset)
    """
    assert abs(train_split + val_split + test_split - 1.0) < 1e-6, "Splits must sum to 1.0"
    
    # Create full dataset
    full_dataset = SignalDataset(clean_signals, noisy_signals)
    
    # Calculate split sizes
    total_size = len(full_dataset)
    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)
    test_size = total_size - train_size - val_size
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size]
    )
    
    print(f"Dataset splits - Train: {train_size}, Val: {val_size}, Test: {test_size}")
    
    return train_dataset, val_dataset, test_dataset


def save_config(config: Dict, filepath: str):
    """Save training configuration to YAML file."""
    with open(filepath, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def load_config(filepath: str) -> Dict:
    """Load training configuration from YAML file."""
    with open(filepath, 'r') as f:
        return yaml.load(f, Loader=yaml.FullLoader)


if __name__ == "__main__":
    # Example usage
    from src.models import ConvAutoencoder
    from src.signal_simulation import generate_gravitational_wave
    from src.noise_models import add_gaussian_noise
    
    print("Generating example training data...")
    
    # Generate sample data
    num_samples = 1000
    signal_length = 4096
    
    clean_signals = []
    noisy_signals = []
    
    for i in tqdm(range(num_samples), desc="Generating signals"):
        # Generate random parameters
        m1 = np.random.uniform(20, 50)
        m2 = np.random.uniform(20, 50)
        snr = np.random.uniform(3, 15)
        
        # Generate signal
        _, clean_signal = generate_gravitational_wave(
            duration=1.0,
            sampling_rate=signal_length,
            m1=m1,
            m2=m2
        )
        
        # Add noise
        noisy_signal = add_gaussian_noise(clean_signal, snr=snr)
        
        clean_signals.append(clean_signal)
        noisy_signals.append(noisy_signal)
    
    clean_signals = np.array(clean_signals)
    noisy_signals = np.array(noisy_signals)
    
    print(f"Generated {num_samples} signals of length {signal_length}")
    
    # Create datasets
    train_dataset, val_dataset, test_dataset = create_datasets(
        clean_signals, noisy_signals
    )
    
    # Initialize model and trainer
    model = ConvAutoencoder(input_length=signal_length)
    trainer = ModelTrainer(model, save_dir='example_models')
    
    # Training configuration
    config = {
        'epochs': 10,  # Small number for demo
        'batch_size': 16,
        'learning_rate': 0.001,
        'optimizer_type': 'adam',
        'criterion_type': 'mse',
        'early_stopping_patience': 5
    }
    
    # Train model
    print("Starting training...")
    history = trainer.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        **config
    )
    
    print("Training completed!")
    print(f"Final train loss: {history['train_losses'][-1]:.6f}")
    print(f"Final val loss: {history['val_losses'][-1]:.6f}")
