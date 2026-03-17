"""Core anomaly detection models for IoT sensor data.

This module provides autoencoder-based anomaly detection models optimized for edge deployment.
Supports various compression techniques including quantization and pruning.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, roc_auc_score

logger = logging.getLogger(__name__)


class PyTorchAutoencoder(nn.Module):
    """PyTorch autoencoder for IoT anomaly detection.
    
    Optimized for edge deployment with configurable compression and quantization.
    
    Args:
        input_dim: Input feature dimension
        hidden_dims: List of hidden layer dimensions
        activation: Activation function ('relu', 'tanh', 'sigmoid')
        dropout: Dropout rate for regularization
        use_batch_norm: Whether to use batch normalization
    """
    
    def __init__(
        self,
        input_dim: int = 1,
        hidden_dims: List[int] = [4, 2],
        activation: str = "relu",
        dropout: float = 0.0,
        use_batch_norm: bool = False,
    ) -> None:
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.activation = activation
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm:
                encoder_layers.append(nn.BatchNorm1d(hidden_dim))
            encoder_layers.append(self._get_activation())
            if dropout > 0:
                encoder_layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
            
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder (reverse of encoder)
        decoder_layers = []
        hidden_dims_reversed = hidden_dims[::-1]
        
        for i, hidden_dim in enumerate(hidden_dims_reversed[1:] + [input_dim]):
            decoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            if i < len(hidden_dims_reversed) - 1:  # Don't apply activation to output
                if use_batch_norm:
                    decoder_layers.append(nn.BatchNorm1d(hidden_dim))
                decoder_layers.append(self._get_activation())
                if dropout > 0:
                    decoder_layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
            
        self.decoder = nn.Sequential(*decoder_layers)
        
    def _get_activation(self) -> nn.Module:
        """Get activation function module."""
        activations = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
        }
        return activations.get(self.activation, nn.ReLU())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through autoencoder."""
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent representation."""
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to reconstruction."""
        return self.decoder(z)


class TensorFlowAutoencoder:
    """TensorFlow/Keras autoencoder for IoT anomaly detection.
    
    Provides compatibility with TensorFlow ecosystem and TFLite export.
    
    Args:
        input_dim: Input feature dimension
        hidden_dims: List of hidden layer dimensions
        activation: Activation function
        dropout: Dropout rate
        use_batch_norm: Whether to use batch normalization
    """
    
    def __init__(
        self,
        input_dim: int = 1,
        hidden_dims: List[int] = [4, 2],
        activation: str = "relu",
        dropout: float = 0.0,
        use_batch_norm: bool = False,
    ) -> None:
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.activation = activation
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        
        self.model = self._build_model()
        self.scaler = MinMaxScaler()
        
    def _build_model(self) -> keras.Model:
        """Build the autoencoder model."""
        # Input layer
        inputs = layers.Input(shape=(self.input_dim,))
        x = inputs
        
        # Encoder
        for hidden_dim in self.hidden_dims:
            x = layers.Dense(hidden_dim)(x)
            if self.use_batch_norm:
                x = layers.BatchNormalization()(x)
            x = layers.Activation(self.activation)(x)
            if self.dropout > 0:
                x = layers.Dropout(self.dropout)(x)
        
        # Decoder
        hidden_dims_reversed = self.hidden_dims[::-1]
        for i, hidden_dim in enumerate(hidden_dims_reversed[1:] + [self.input_dim]):
            x = layers.Dense(hidden_dim)(x)
            if i < len(hidden_dims_reversed) - 1:  # Don't apply activation to output
                if self.use_batch_norm:
                    x = layers.BatchNormalization()(x)
                x = layers.Activation(self.activation)(x)
                if self.dropout > 0:
                    x = layers.Dropout(self.dropout)(x)
        
        # Create model
        model = keras.Model(inputs, x, name="autoencoder")
        return model
    
    def compile(self, optimizer: str = "adam", loss: str = "mse") -> None:
        """Compile the model."""
        self.model.compile(optimizer=optimizer, loss=loss)
    
    def fit(
        self,
        x: np.ndarray,
        y: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: int = 32,
        validation_split: float = 0.2,
        verbose: int = 1,
    ) -> keras.callbacks.History:
        """Train the autoencoder."""
        if y is None:
            y = x  # Autoencoder learns to reconstruct input
            
        return self.model.fit(
            x, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=verbose,
        )
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Make predictions."""
        return self.model.predict(x, verbose=0)
    
    def encode(self, x: np.ndarray) -> np.ndarray:
        """Encode input to latent representation."""
        # Extract encoder part
        encoder_layers = self.model.layers[:len(self.hidden_dims) * 3]  # Approximate
        encoder_input = layers.Input(shape=(self.input_dim,))
        encoder_output = encoder_input
        
        for layer in encoder_layers:
            encoder_output = layer(encoder_output)
            
        encoder_model = keras.Model(encoder_input, encoder_output)
        return encoder_model.predict(x, verbose=0)


class AnomalyDetector:
    """Main anomaly detection class with multiple algorithms and edge optimization.
    
    Supports both PyTorch and TensorFlow backends with compression techniques.
    
    Args:
        model_type: Type of model ('pytorch' or 'tensorflow')
        input_dim: Input feature dimension
        hidden_dims: Hidden layer dimensions
        threshold_method: Method for anomaly threshold ('statistical', 'percentile', 'iqr')
        threshold_value: Threshold value or percentile
        device: Device for PyTorch models ('cpu', 'cuda', 'auto')
    """
    
    def __init__(
        self,
        model_type: str = "tensorflow",
        input_dim: int = 1,
        hidden_dims: List[int] = [4, 2],
        threshold_method: str = "statistical",
        threshold_value: float = 3.0,
        device: str = "auto",
    ) -> None:
        self.model_type = model_type.lower()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.threshold_method = threshold_method
        self.threshold_value = threshold_value
        
        # Set device for PyTorch
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        logger.info(f"Using device: {self.device}")
        
        # Initialize model
        if self.model_type == "pytorch":
            self.model = PyTorchAutoencoder(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
            ).to(self.device)
        elif self.model_type == "tensorflow":
            self.model = TensorFlowAutoencoder(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
            
        self.scaler = MinMaxScaler()
        self.threshold = None
        self.is_fitted = False
        
    def fit(self, data: np.ndarray, **kwargs) -> AnomalyDetector:
        """Train the anomaly detection model.
        
        Args:
            data: Training data (normal samples)
            **kwargs: Additional training parameters
            
        Returns:
            Self for method chaining
        """
        logger.info(f"Training {self.model_type} autoencoder on {len(data)} samples")
        
        # Normalize data
        data_scaled = self.scaler.fit_transform(data)
        
        if self.model_type == "pytorch":
            self._fit_pytorch(data_scaled, **kwargs)
        else:
            self._fit_tensorflow(data_scaled, **kwargs)
            
        # Calculate threshold on training data
        self._calculate_threshold(data_scaled)
        self.is_fitted = True
        
        logger.info(f"Training completed. Threshold: {self.threshold:.4f}")
        return self
    
    def _fit_pytorch(self, data: np.ndarray, epochs: int = 100, batch_size: int = 32) -> None:
        """Train PyTorch model."""
        # Convert to tensors
        data_tensor = torch.FloatTensor(data).to(self.device)
        dataset = TensorDataset(data_tensor, data_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Setup training
        optimizer = torch.optim.Adam(self.model.parameters())
        criterion = nn.MSELoss()
        
        # Training loop
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()
                reconstructed = self.model(batch_x)
                loss = criterion(reconstructed, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
            if epoch % 20 == 0:
                logger.info(f"Epoch {epoch}, Loss: {total_loss/len(dataloader):.4f}")
    
    def _fit_tensorflow(self, data: np.ndarray, epochs: int = 100, batch_size: int = 32) -> None:
        """Train TensorFlow model."""
        self.model.compile(optimizer="adam", loss="mse")
        self.model.fit(
            data, data,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=0,
        )
    
    def _calculate_threshold(self, data: np.ndarray) -> None:
        """Calculate anomaly threshold based on reconstruction errors."""
        # Get reconstruction errors
        errors = self._get_reconstruction_errors(data)
        
        if self.threshold_method == "statistical":
            # Mean + threshold_value * std
            self.threshold = np.mean(errors) + self.threshold_value * np.std(errors)
        elif self.threshold_method == "percentile":
            # Percentile-based threshold
            self.threshold = np.percentile(errors, self.threshold_value)
        elif self.threshold_method == "iqr":
            # Interquartile range method
            q1, q3 = np.percentile(errors, [25, 75])
            iqr = q3 - q1
            self.threshold = q3 + self.threshold_value * iqr
        else:
            raise ValueError(f"Unknown threshold method: {self.threshold_method}")
    
    def _get_reconstruction_errors(self, data: np.ndarray) -> np.ndarray:
        """Calculate reconstruction errors for given data."""
        if self.model_type == "pytorch":
            with torch.no_grad():
                data_tensor = torch.FloatTensor(data).to(self.device)
                reconstructed = self.model(data_tensor)
                errors = torch.abs(data_tensor - reconstructed).cpu().numpy()
                return np.mean(errors, axis=1)  # Mean error per sample
        else:
            reconstructed = self.model.predict(data, verbose=0)
            errors = np.abs(data - reconstructed)
            return np.mean(errors, axis=1)  # Mean error per sample
    
    def predict(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Detect anomalies in given data.
        
        Args:
            data: Input data to analyze
            
        Returns:
            Tuple of (anomaly_flags, reconstruction_errors)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
            
        # Normalize data
        data_scaled = self.scaler.transform(data)
        
        # Get reconstruction errors
        errors = self._get_reconstruction_errors(data_scaled)
        
        # Flag anomalies
        anomaly_flags = errors > self.threshold
        
        return anomaly_flags, errors
    
    def evaluate(self, data: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance on labeled data.
        
        Args:
            data: Input data
            labels: True anomaly labels (1 for anomaly, 0 for normal)
            
        Returns:
            Dictionary of evaluation metrics
        """
        anomaly_flags, errors = self.predict(data)
        
        # Calculate metrics
        accuracy = np.mean(anomaly_flags == labels)
        precision = np.sum((anomaly_flags == 1) & (labels == 1)) / np.sum(anomaly_flags == 1)
        recall = np.sum((anomaly_flags == 1) & (labels == 1)) / np.sum(labels == 1)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # AUC using reconstruction errors as scores
        try:
            auc = roc_auc_score(labels, errors)
        except ValueError:
            auc = 0.5  # Default for edge cases
            
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "auc": auc,
            "threshold": self.threshold,
        }
    
    def get_model_size(self) -> Dict[str, Any]:
        """Get model size information for edge deployment analysis."""
        if self.model_type == "pytorch":
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            # Estimate model size in MB (assuming float32)
            model_size_mb = total_params * 4 / (1024 * 1024)
            
            return {
                "total_parameters": total_params,
                "trainable_parameters": trainable_params,
                "model_size_mb": model_size_mb,
                "model_type": "pytorch",
            }
        else:
            # TensorFlow model size estimation
            total_params = self.model.count_params()
            model_size_mb = total_params * 4 / (1024 * 1024)
            
            return {
                "total_parameters": total_params,
                "trainable_parameters": total_params,
                "model_size_mb": model_size_mb,
                "model_type": "tensorflow",
            }
