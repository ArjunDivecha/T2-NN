import os
import logging

# Paths
DATA_DIR = 'data/'
OUTPUT_DIR = 'outputs/'
PLOTS_DIR = 'outputs/plots/'

# Data specifications (updated from analysis)
N_FACTORS = 83  # Updated from 85 based on actual data files
FACTOR_NAMES = None  # Will be loaded from data files

# Model defaults (updated from hyperparameter tuning)
DEFAULT_CONFIG = {
    'hidden_sizes': [512, 256],
    'learning_rate': 0.001,
    'dropout_rate': 0.4,
    'batch_size': 32,
    'weight_decay': 1e-05
}

# Training settings
TRAINING_CONFIG = {
    'n_epochs': 100,
    'early_stopping_patience': 10,
    'val_split': 0.2,
    'random_seed': 42
}

# Backtest settings
BACKTEST_CONFIG = {
    'window_size': 60,
    'start_date': None,  # Use all available data
    'n_jobs': 8
}

# Hyperparameter grid for tuning
HYPERPARAM_GRID = {
    'hidden_sizes': [
        [256],
        [512, 256],
        [1024, 512, 256]
    ],
    'learning_rate': [1e-4, 1e-3, 1e-2],
    'dropout_rate': [0.0, 0.2, 0.4],
    'batch_size': [16, 32],
    'weight_decay': [0, 1e-5, 1e-4]
}

# Device configuration
def get_device():
    """Get the best available device (MPS > CPU)"""
    import torch
    if torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

# Logging configuration
def setup_logging(log_file='outputs/training.log'):
    """Setup logging configuration"""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)