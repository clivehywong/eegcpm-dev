"""Deep learning models for EEG prediction (optional dependency)."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def check_torch_available() -> bool:
    """Check if PyTorch is available."""
    try:
        import torch
        return True
    except ImportError:
        return False


def check_tensorflow_available() -> bool:
    """Check if TensorFlow is available."""
    try:
        import tensorflow
        return True
    except ImportError:
        return False


class SimpleCNN:
    """
    Simple 1D CNN for EEG feature prediction.

    Requires PyTorch.
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int] = [64, 32],
        dropout: float = 0.3,
        learning_rate: float = 0.001,
        epochs: int = 100,
    ):
        if not check_torch_available():
            raise ImportError("PyTorch is required for SimpleCNN")

        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.model = None
        self.device = None

    def _build_model(self, output_size: int = 1):
        """Build the CNN model."""
        import torch
        import torch.nn as nn

        class CNN1D(nn.Module):
            def __init__(self, input_size, hidden_sizes, dropout, output_size):
                super().__init__()
                layers = []

                # Reshape for 1D conv
                in_channels = 1
                for i, hidden in enumerate(hidden_sizes):
                    layers.append(nn.Conv1d(in_channels, hidden, kernel_size=3, padding=1))
                    layers.append(nn.ReLU())
                    layers.append(nn.BatchNorm1d(hidden))
                    layers.append(nn.Dropout(dropout))
                    in_channels = hidden

                self.conv = nn.Sequential(*layers)
                self.fc = nn.Linear(hidden_sizes[-1] * input_size, output_size)

            def forward(self, x):
                # x: (batch, features)
                x = x.unsqueeze(1)  # (batch, 1, features)
                x = self.conv(x)  # (batch, hidden, features)
                x = x.view(x.size(0), -1)  # Flatten
                return self.fc(x)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CNN1D(
            self.input_size,
            self.hidden_sizes,
            self.dropout,
            output_size,
        ).to(self.device)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SimpleCNN":
        """Train the model."""
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        self._build_model(output_size=1)

        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).reshape(-1, 1).to(self.device)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()

        self.model.train()
        for epoch in range(self.epochs):
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        import torch

        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            predictions = self.model(X_tensor).cpu().numpy()
        return predictions.flatten()


class SimpleLSTM:
    """
    Simple LSTM for sequential EEG data.

    Requires PyTorch.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        learning_rate: float = 0.001,
        epochs: int = 100,
    ):
        if not check_torch_available():
            raise ImportError("PyTorch is required for SimpleLSTM")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.model = None
        self.device = None

    def _build_model(self, output_size: int = 1):
        """Build the LSTM model."""
        import torch
        import torch.nn as nn

        class LSTMModel(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers, dropout, output_size):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size=1,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=dropout if num_layers > 1 else 0,
                )
                self.fc = nn.Linear(hidden_size, output_size)

            def forward(self, x):
                # x: (batch, seq_len, 1)
                lstm_out, _ = self.lstm(x)
                # Use last hidden state
                return self.fc(lstm_out[:, -1, :])

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = LSTMModel(
            self.input_size,
            self.hidden_size,
            self.num_layers,
            self.dropout,
            output_size,
        ).to(self.device)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SimpleLSTM":
        """Train the model."""
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        self._build_model(output_size=1)

        # Reshape for LSTM: (batch, seq_len, features)
        X_reshaped = X.reshape(X.shape[0], X.shape[1], 1)
        X_tensor = torch.FloatTensor(X_reshaped).to(self.device)
        y_tensor = torch.FloatTensor(y).reshape(-1, 1).to(self.device)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()

        self.model.train()
        for epoch in range(self.epochs):
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        import torch

        self.model.eval()
        with torch.no_grad():
            X_reshaped = X.reshape(X.shape[0], X.shape[1], 1)
            X_tensor = torch.FloatTensor(X_reshaped).to(self.device)
            predictions = self.model(X_tensor).cpu().numpy()
        return predictions.flatten()


class NeuralModelFactory:
    """Factory for creating neural network models."""

    @staticmethod
    def create(
        model_type: str,
        input_size: int,
        **kwargs,
    ):
        """
        Create a neural network model.

        Args:
            model_type: Type of model (cnn, lstm)
            input_size: Number of input features
            **kwargs: Model parameters

        Returns:
            Neural network model
        """
        if model_type == "cnn":
            return SimpleCNN(input_size, **kwargs)
        elif model_type == "lstm":
            return SimpleLSTM(input_size, **kwargs)
        else:
            raise ValueError(f"Unknown neural model type: {model_type}")
