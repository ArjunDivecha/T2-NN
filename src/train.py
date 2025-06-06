import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import logging
import time
import copy
from typing import Dict, Tuple, Optional, List
from sklearn.model_selection import train_test_split
import os

from .model import SimpleNN, top5_return_loss, top5_return_loss_eval, calculate_top5_metrics, create_model, get_optimizer
from .config import get_device, TRAINING_CONFIG, OUTPUT_DIR

logger = logging.getLogger(__name__)

class EarlyStopping:
    """Early stopping utility to prevent overfitting."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, restore_best_weights: bool = True):
        """
        Args:
            patience: Number of epochs with no improvement after which training will be stopped
            min_delta: Minimum change in the monitored quantity to qualify as an improvement
            restore_best_weights: Whether to restore model weights from the epoch with the best value
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        """
        Check if training should stop.
        
        Args:
            val_loss: Current validation loss
            model: Model to potentially save weights from
            
        Returns:
            True if training should stop, False otherwise
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
            
        should_stop = self.counter >= self.patience
        
        if should_stop and self.restore_best_weights and self.best_weights is not None:
            model.load_state_dict(self.best_weights)
            logger.info(f"Early stopping triggered. Restored best weights from {self.patience - self.counter} epochs ago.")
            
        return should_stop

def create_data_loaders(train_X: np.ndarray, train_y: np.ndarray, 
                       val_X: Optional[np.ndarray] = None, val_y: Optional[np.ndarray] = None,
                       batch_size: int = 32, random_seed: int = 42) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation data loaders.
    
    Args:
        train_X: Training features
        train_y: Training targets
        val_X: Validation features (optional)
        val_y: Validation targets (optional)
        batch_size: Batch size for data loaders
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Convert to tensors
    train_X_tensor = torch.FloatTensor(train_X)
    train_y_tensor = torch.FloatTensor(train_y)
    
    # If no validation data provided, use ALL training data (no split)
    if val_X is None or val_y is None:
        # Use all training data for both training and validation
        # This ensures we train on the full 60-month window
        val_X_tensor = train_X_tensor.clone()
        val_y_tensor = train_y_tensor.clone()
        
        logger.info(f"Using ALL training data (no validation split): train={train_X.shape[0]}, val={train_X.shape[0]}")
    else:
        val_X_tensor = torch.FloatTensor(val_X)
        val_y_tensor = torch.FloatTensor(val_y)
        logger.info(f"Using provided validation data: train={train_X.shape[0]}, val={val_X.shape[0]}")
    
    # Create datasets
    train_dataset = TensorDataset(train_X_tensor, train_y_tensor)
    val_dataset = TensorDataset(val_X_tensor, val_y_tensor)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

def train_epoch(model: SimpleNN, train_loader: DataLoader, optimizer: torch.optim.Optimizer, 
                device: torch.device) -> Dict[str, float]:
    """
    Train the model for one epoch.
    
    Args:
        model: Neural network model
        train_loader: Training data loader
        optimizer: Optimizer
        device: Device to run on
        
    Returns:
        Dictionary with training metrics
    """
    model.train()
    total_loss = 0.0
    total_samples = 0
    all_returns = []
    all_hit_rates = []
    
    for batch_idx, (data, targets) in enumerate(train_loader):
        data, targets = data.to(device), targets.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        predictions = model(data)
        
        # Calculate loss (differentiable version for training)
        loss = top5_return_loss(predictions, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item() * data.size(0)
        total_samples += data.size(0)
        
        # Calculate metrics for monitoring
        with torch.no_grad():
            metrics = calculate_top5_metrics(predictions, targets)
            all_returns.append(metrics['avg_top5_return'])
            all_hit_rates.append(metrics['hit_rate'])
    
    avg_loss = total_loss / total_samples
    avg_return = np.mean(all_returns)
    avg_hit_rate = np.mean(all_hit_rates)
    
    return {
        'loss': avg_loss,
        'avg_top5_return': avg_return,
        'hit_rate': avg_hit_rate
    }

def validate_epoch(model: SimpleNN, val_loader: DataLoader, device: torch.device) -> Dict[str, float]:
    """
    Validate the model for one epoch.
    
    Args:
        model: Neural network model
        val_loader: Validation data loader
        device: Device to run on
        
    Returns:
        Dictionary with validation metrics
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_returns = []
    all_hit_rates = []
    
    with torch.no_grad():
        for data, targets in val_loader:
            data, targets = data.to(device), targets.to(device)
            
            # Forward pass
            predictions = model(data)
            
            # Calculate loss (non-differentiable evaluation version)
            loss = top5_return_loss_eval(predictions, targets)
            
            # Track metrics
            total_loss += loss.item() * data.size(0)
            total_samples += data.size(0)
            
            # Calculate additional metrics
            metrics = calculate_top5_metrics(predictions, targets)
            all_returns.append(metrics['avg_top5_return'])
            all_hit_rates.append(metrics['hit_rate'])
    
    avg_loss = total_loss / total_samples
    avg_return = np.mean(all_returns)
    avg_hit_rate = np.mean(all_hit_rates)
    
    return {
        'loss': avg_loss,
        'avg_top5_return': avg_return,
        'hit_rate': avg_hit_rate
    }

def train_model(train_X: np.ndarray, train_y: np.ndarray, 
                val_X: Optional[np.ndarray] = None, val_y: Optional[np.ndarray] = None,
                config: Optional[Dict] = None) -> Tuple[SimpleNN, Dict]:
    """
    Train a neural network model with the given data and configuration.
    
    Args:
        train_X: Training features of shape (n_samples, n_factors)
        train_y: Training targets of shape (n_samples, n_factors)
        val_X: Validation features (optional)
        val_y: Validation targets (optional)
        config: Training configuration dictionary
        
    Returns:
        Tuple of (trained_model, training_history)
    """
    # Use default config if none provided
    if config is None:
        from .config import DEFAULT_CONFIG
        config = DEFAULT_CONFIG.copy()
        config.update(TRAINING_CONFIG)
    
    # Get device
    device = get_device()
    logger.info(f"Training on device: {device}")
    
    # Set random seed for reproducibility
    torch.manual_seed(config.get('random_seed', 42))
    if device.type == 'mps':
        torch.mps.manual_seed(config.get('random_seed', 42))
    np.random.seed(config.get('random_seed', 42))
    
    # Create model
    model_config = {
        'input_size': train_X.shape[1],
        'hidden_sizes': config.get('hidden_sizes', [512, 256]),
        'dropout_rate': config.get('dropout_rate', 0.2)
    }
    model = create_model(model_config, device)
    
    # Create optimizer
    optimizer = get_optimizer(model, config)
    
    # Create data loaders
    batch_size = config.get('batch_size', 32)
    random_seed = config.get('random_seed', 42)
    train_loader, val_loader = create_data_loaders(
        train_X, train_y, val_X, val_y, batch_size, random_seed
    )
    
    # Setup early stopping
    patience = config.get('early_stopping_patience', 10)
    early_stopping = EarlyStopping(patience=patience, restore_best_weights=True)
    
    # Training loop
    n_epochs = config.get('n_epochs', 100)
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_return': [],
        'val_return': [],
        'train_hit_rate': [],
        'val_hit_rate': [],
        'epoch_times': []
    }
    
    logger.info(f"Starting training for {n_epochs} epochs with batch size {batch_size}")
    start_time = time.time()
    
    for epoch in range(n_epochs):
        epoch_start = time.time()
        
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, device)
        
        # Validate
        val_metrics = validate_epoch(model, val_loader, device)
        
        epoch_time = time.time() - epoch_start
        
        # Record history
        history['train_loss'].append(train_metrics['loss'])
        history['val_loss'].append(val_metrics['loss'])
        history['train_return'].append(train_metrics['avg_top5_return'])
        history['val_return'].append(val_metrics['avg_top5_return'])
        history['train_hit_rate'].append(train_metrics['hit_rate'])
        history['val_hit_rate'].append(val_metrics['hit_rate'])
        history['epoch_times'].append(epoch_time)
        
        # Log progress
        if epoch % 10 == 0 or epoch < 5:
            logger.info(
                f"Epoch {epoch+1}/{n_epochs} - "
                f"Train Loss: {train_metrics['loss']:.6f}, "
                f"Val Loss: {val_metrics['loss']:.6f}, "
                f"Val Return: {val_metrics['avg_top5_return']:.4f}, "
                f"Val Hit Rate: {val_metrics['hit_rate']:.3f}, "
                f"Time: {epoch_time:.2f}s"
            )
        
        # Early stopping check
        if early_stopping(val_metrics['loss'], model):
            logger.info(f"Early stopping at epoch {epoch+1}")
            break
    
    total_time = time.time() - start_time
    final_epoch = len(history['train_loss'])
    
    logger.info(f"Training completed in {total_time:.2f}s ({final_epoch} epochs)")
    logger.info(f"Best validation loss: {min(history['val_loss']):.6f}")
    logger.info(f"Best validation return: {max(history['val_return']):.4f}")
    
    # Add summary to history
    history['total_time'] = total_time
    history['final_epoch'] = final_epoch
    history['best_val_loss'] = min(history['val_loss'])
    history['best_val_return'] = max(history['val_return'])
    
    return model, history

def save_model_checkpoint(model: SimpleNN, filepath: str, config: Dict, history: Dict):
    """
    Save model checkpoint with configuration and training history.
    
    Args:
        model: Trained model
        filepath: Path to save checkpoint
        config: Model configuration
        history: Training history
    """
    import os
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_config': model.get_model_info(),
        'training_config': config,
        'training_history': history
    }
    
    torch.save(checkpoint, filepath)
    logger.info(f"Model checkpoint saved to {filepath}")

def load_model_checkpoint(filepath: str, device: torch.device) -> Tuple[SimpleNN, Dict, Dict]:
    """
    Load model checkpoint.
    
    Args:
        filepath: Path to checkpoint file
        device: Device to load model on
        
    Returns:
        Tuple of (model, config, history)
    """
    # Use weights_only=False for trusted checkpoints (PyTorch 2.6+ compatibility)
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    
    # Create model with saved configuration
    model_config = checkpoint['model_config']
    model = SimpleNN(
        input_size=model_config['input_size'],
        hidden_sizes=model_config['hidden_sizes'],
        dropout_rate=model_config['dropout_rate']
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    logger.info(f"Model checkpoint loaded from {filepath}")
    
    return model, checkpoint['training_config'], checkpoint['training_history']

def save_monthly_forecasts(model: SimpleNN, X: np.ndarray, date: pd.Timestamp, 
                          factor_names: List[str], device: torch.device,
                          output_file: str = None) -> pd.DataFrame:
    """
    Generate and save monthly factor forecasts in T60.xlsx format.
    
    Args:
        model: Trained neural network model
        X: Input features for prediction (1 sample, n_factors)
        date: Date for this prediction
        factor_names: List of factor column names
        device: Device to run prediction on
        output_file: Optional output Excel file path
        
    Returns:
        DataFrame with Date and all factor forecasts
    """
    model.eval()
    
    # Ensure X has the right shape (1, n_factors)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    
    # Convert to tensor and predict
    X_tensor = torch.FloatTensor(X).to(device)
    
    with torch.no_grad():
        predictions = model(X_tensor)
        forecasts = predictions.cpu().numpy().flatten()
    
    # Create DataFrame matching T60.xlsx format
    forecast_data = {'Date': [date]}
    for i, factor_name in enumerate(factor_names):
        forecast_data[factor_name] = [forecasts[i]]
    
    forecast_df = pd.DataFrame(forecast_data)
    
    # Save to Excel if output file specified
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        forecast_df.to_excel(output_file, index=False)
        logger.info(f"Monthly forecasts saved to {output_file}")
    
    return forecast_df

def save_rolling_forecasts(forecasts_list: List[pd.DataFrame], 
                          output_file: str = None) -> pd.DataFrame:
    """
    Combine multiple monthly forecasts into a single Excel file.
    
    Args:
        forecasts_list: List of monthly forecast DataFrames
        output_file: Output Excel file path
        
    Returns:
        Combined DataFrame with all monthly forecasts
    """
    # Combine all forecasts
    all_forecasts = pd.concat(forecasts_list, ignore_index=True)
    
    # Sort by date
    all_forecasts = all_forecasts.sort_values('Date').reset_index(drop=True)
    
    # Save to Excel if output file specified
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        all_forecasts.to_excel(output_file, index=False)
        logger.info(f"All rolling forecasts saved to {output_file} ({len(all_forecasts)} months)")
    
    return all_forecasts