"""
Neural Network Models for Signal Denoising

This module contains various autoencoder architectures for denoising astrophysical signals.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Optional


class ConvAutoencoder(nn.Module):
    """
    1D Convolutional Autoencoder for signal denoising.
    """
    
    def __init__(
        self,
        input_length: int = 4096,
        hidden_dims: List[int] = [256, 128, 64, 32],
        kernel_size: int = 3,
        activation: str = 'relu',
        dropout_rate: float = 0.1
    ):
        """
        Initialize the Convolutional Autoencoder.
        
        Parameters:
        -----------
        input_length : int
            Length of input signal
        hidden_dims : list
            Hidden layer dimensions
        kernel_size : int
            Convolutional kernel size
        activation : str
            Activation function ('relu', 'leaky_relu', 'elu')
        dropout_rate : float
            Dropout rate for regularization
        """
        super(ConvAutoencoder, self).__init__()
        
        self.input_length = input_length
        self.hidden_dims = hidden_dims
        self.activation = self._get_activation(activation)
        
        # Encoder layers
        encoder_layers = []
        in_channels = 1
        
        for dim in hidden_dims:
            encoder_layers.extend([
                nn.Conv1d(in_channels, dim, kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(dim),
                self.activation,
                nn.MaxPool1d(2),
                nn.Dropout(dropout_rate)
            ])
            in_channels = dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Calculate compressed length
        compressed_length = input_length
        for _ in hidden_dims:
            compressed_length = compressed_length // 2
        
        # Decoder layers
        decoder_layers = []
        hidden_dims_reversed = hidden_dims[::-1]
        
        for i, dim in enumerate(hidden_dims_reversed[:-1]):
            next_dim = hidden_dims_reversed[i + 1]
            decoder_layers.extend([
                nn.ConvTranspose1d(dim, next_dim, kernel_size*2, stride=2, padding=kernel_size//2),
                nn.BatchNorm1d(next_dim),
                self.activation,
                nn.Dropout(dropout_rate)
            ])
        
        # Final layer to reconstruct signal
        decoder_layers.extend([
            nn.ConvTranspose1d(hidden_dims_reversed[-1], 1, kernel_size*2, stride=2, padding=kernel_size//2),
            nn.Tanh()  # Output between -1 and 1
        ])
        
        self.decoder = nn.Sequential(*decoder_layers)
        
    def _get_activation(self, activation: str):
        """Get activation function."""
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'leaky_relu':
            return nn.LeakyReLU(0.2)
        elif activation == 'elu':
            return nn.ELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the autoencoder.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, 1, sequence_length)
        
        Returns:
        --------
        torch.Tensor
            Reconstructed signal
        """
        # Encoder
        encoded = self.encoder(x)
        
        # Decoder
        decoded = self.decoder(encoded)
        
        # Ensure output has same length as input
        if decoded.size(-1) != x.size(-1):
            decoded = F.interpolate(decoded, size=x.size(-1), mode='linear', align_corners=False)
        
        return decoded


class LSTMAutoencoder(nn.Module):
    """
    LSTM-based Autoencoder for sequential signal denoising.
    """
    
    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout_rate: float = 0.1,
        bidirectional: bool = True
    ):
        """
        Initialize the LSTM Autoencoder.
        
        Parameters:
        -----------
        input_size : int
            Input feature size
        hidden_size : int
            Hidden state size
        num_layers : int
            Number of LSTM layers
        dropout_rate : float
            Dropout rate
        bidirectional : bool
            Whether to use bidirectional LSTM
        """
        super(LSTMAutoencoder, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # Encoder LSTM
        self.encoder_lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Decoder LSTM
        lstm_output_size = hidden_size * (2 if bidirectional else 1)
        self.decoder_lstm = nn.LSTM(
            input_size=lstm_output_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Output layer
        self.output_layer = nn.Linear(lstm_output_size, input_size)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the LSTM autoencoder.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, sequence_length, input_size)
        
        Returns:
        --------
        torch.Tensor
            Reconstructed signal
        """
        batch_size, seq_len, _ = x.size()
        
        # Encoder
        encoded, (hidden, cell) = self.encoder_lstm(x)
        encoded = self.dropout(encoded)
        
        # Use the last encoded state as context
        if self.bidirectional:
            # Concatenate forward and backward hidden states
            context = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            context = hidden[-1]
        
        # Repeat context for each time step
        context = context.unsqueeze(1).repeat(1, seq_len, 1)
        
        # Decoder
        decoded, _ = self.decoder_lstm(context)
        decoded = self.dropout(decoded)
        
        # Output projection
        output = self.output_layer(decoded)
        
        return output


class UNetAutoencoder(nn.Module):
    """
    U-Net style autoencoder with skip connections for signal denoising.
    """
    
    def __init__(
        self,
        input_length: int = 4096,
        base_channels: int = 64,
        depth: int = 4,
        kernel_size: int = 3
    ):
        """
        Initialize the U-Net Autoencoder.
        
        Parameters:
        -----------
        input_length : int
            Length of input signal
        base_channels : int
            Base number of channels
        depth : int
            Network depth (number of downsampling/upsampling steps)
        kernel_size : int
            Convolutional kernel size
        """
        super(UNetAutoencoder, self).__init__()
        
        self.depth = depth
        
        # Encoder (downsampling path)
        self.encoder_blocks = nn.ModuleList()
        in_channels = 1
        
        for i in range(depth):
            out_channels = base_channels * (2 ** i)
            self.encoder_blocks.append(
                self._make_conv_block(in_channels, out_channels, kernel_size)
            )
            in_channels = out_channels
        
        # Bottleneck
        self.bottleneck = self._make_conv_block(
            in_channels, in_channels * 2, kernel_size
        )
        
        # Decoder (upsampling path)
        self.decoder_blocks = nn.ModuleList()
        in_channels = in_channels * 2
        
        for i in range(depth):
            skip_channels = base_channels * (2 ** (depth - 1 - i))
            out_channels = skip_channels
            
            self.decoder_blocks.append(
                self._make_upconv_block(in_channels + skip_channels, out_channels, kernel_size)
            )
            in_channels = out_channels
        
        # Final output layer
        self.final_conv = nn.Conv1d(base_channels, 1, 1)
        
    def _make_conv_block(self, in_channels: int, out_channels: int, kernel_size: int) -> nn.Module:
        """Create a convolutional block."""
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def _make_upconv_block(self, in_channels: int, out_channels: int, kernel_size: int) -> nn.Module:
        """Create an upsampling convolutional block."""
        return nn.Sequential(
            nn.ConvTranspose1d(in_channels, out_channels, kernel_size*2, stride=2, padding=kernel_size//2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the U-Net autoencoder.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, 1, sequence_length)
        
        Returns:
        --------
        torch.Tensor
            Reconstructed signal
        """
        # Encoder path with skip connections storage
        skip_connections = []
        current = x
        
        for encoder_block in self.encoder_blocks:
            current = encoder_block(current)
            skip_connections.append(current)
            current = F.max_pool1d(current, 2)
        
        # Bottleneck
        current = self.bottleneck(current)
        
        # Decoder path with skip connections
        for i, decoder_block in enumerate(self.decoder_blocks):
            # Upsample
            current = F.interpolate(current, scale_factor=2, mode='linear', align_corners=False)
            
            # Concatenate with corresponding skip connection
            skip = skip_connections[-(i+1)]
            if current.size(-1) != skip.size(-1):
                current = F.interpolate(current, size=skip.size(-1), mode='linear', align_corners=False)
            
            current = torch.cat([current, skip], dim=1)
            current = decoder_block(current)
        
        # Final output
        output = self.final_conv(current)
        
        # Ensure output has same length as input
        if output.size(-1) != x.size(-1):
            output = F.interpolate(output, size=x.size(-1), mode='linear', align_corners=False)
        
        return torch.tanh(output)


class WaveNetAutoencoder(nn.Module):
    """
    WaveNet-inspired autoencoder with dilated convolutions.
    """
    
    def __init__(
        self,
        input_length: int = 4096,
        residual_channels: int = 32,
        skip_channels: int = 32,
        dilation_channels: int = 32,
        num_blocks: int = 3,
        num_layers_per_block: int = 10
    ):
        """
        Initialize the WaveNet Autoencoder.
        
        Parameters:
        -----------
        input_length : int
            Length of input signal
        residual_channels : int
            Number of residual channels
        skip_channels : int
            Number of skip channels
        dilation_channels : int
            Number of dilation channels
        num_blocks : int
            Number of residual blocks
        num_layers_per_block : int
            Number of layers per block
        """
        super(WaveNetAutoencoder, self).__init__()
        
        self.num_blocks = num_blocks
        self.num_layers_per_block = num_layers_per_block
        
        # Initial convolution
        self.start_conv = nn.Conv1d(1, residual_channels, 1)
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList()
        for block in range(num_blocks):
            for layer in range(num_layers_per_block):
                dilation = 2 ** layer
                self.residual_blocks.append(
                    WaveNetResidualBlock(
                        residual_channels,
                        skip_channels,
                        dilation_channels,
                        dilation
                    )
                )
        
        # Output layers
        self.end_conv_1 = nn.Conv1d(skip_channels, skip_channels, 1)
        self.end_conv_2 = nn.Conv1d(skip_channels, 1, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through WaveNet autoencoder."""
        current = self.start_conv(x)
        skip_connections = []
        
        for block in self.residual_blocks:
            current, skip = block(current)
            skip_connections.append(skip)
        
        # Sum all skip connections
        skip_sum = torch.stack(skip_connections, dim=0).sum(dim=0)
        
        # Output layers
        output = F.relu(skip_sum)
        output = F.relu(self.end_conv_1(output))
        output = self.end_conv_2(output)
        
        return torch.tanh(output)


class WaveNetResidualBlock(nn.Module):
    """Residual block for WaveNet architecture."""
    
    def __init__(
        self,
        residual_channels: int,
        skip_channels: int,
        dilation_channels: int,
        dilation: int
    ):
        super(WaveNetResidualBlock, self).__init__()
        
        self.dilated_conv = nn.Conv1d(
            residual_channels,
            dilation_channels,
            2,
            dilation=dilation,
            padding=dilation
        )
        
        self.conv_1x1 = nn.Conv1d(dilation_channels, residual_channels, 1)
        self.skip_conv = nn.Conv1d(dilation_channels, skip_channels, 1)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through residual block."""
        residual = x
        
        # Dilated convolution
        conv_out = self.dilated_conv(x)
        
        # Gated activation
        tanh_out = torch.tanh(conv_out)
        sigmoid_out = torch.sigmoid(conv_out)
        gated = tanh_out * sigmoid_out
        
        # 1x1 convolutions
        skip = self.skip_conv(gated)
        residual_out = self.conv_1x1(gated)
        
        # Residual connection
        output = residual + residual_out
        
        return output, skip


def get_model(model_type: str, **kwargs) -> nn.Module:
    """
    Factory function to create model instances.
    
    Parameters:
    -----------
    model_type : str
        Type of model ('conv', 'lstm', 'unet', 'wavenet')
    **kwargs
        Model-specific parameters
    
    Returns:
    --------
    nn.Module
        Initialized model
    """
    if model_type == 'conv':
        return ConvAutoencoder(**kwargs)
    elif model_type == 'lstm':
        return LSTMAutoencoder(**kwargs)
    elif model_type == 'unet':
        return UNetAutoencoder(**kwargs)
    elif model_type == 'wavenet':
        return WaveNetAutoencoder(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Example usage and model testing
    print("Testing neural network models...")
    
    # Test ConvAutoencoder
    model = ConvAutoencoder(input_length=4096)
    x = torch.randn(8, 1, 4096)  # Batch of 8 signals
    output = model(x)
    print(f"ConvAutoencoder - Input: {x.shape}, Output: {output.shape}")
    
    # Test LSTMAutoencoder
    model = LSTMAutoencoder()
    x = torch.randn(8, 4096, 1)  # Batch of 8 sequences
    output = model(x)
    print(f"LSTMAutoencoder - Input: {x.shape}, Output: {output.shape}")
    
    # Test UNetAutoencoder
    model = UNetAutoencoder(input_length=4096)
    x = torch.randn(8, 1, 4096)
    output = model(x)
    print(f"UNetAutoencoder - Input: {x.shape}, Output: {output.shape}")
    
    # Test WaveNetAutoencoder
    model = WaveNetAutoencoder(input_length=4096)
    x = torch.randn(8, 1, 4096)
    output = model(x)
    print(f"WaveNetAutoencoder - Input: {x.shape}, Output: {output.shape}")
    
    print("All models tested successfully!")
