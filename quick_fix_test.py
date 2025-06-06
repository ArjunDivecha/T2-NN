#!/usr/bin/env python3
"""
Quick test of fixed temporal validation with a limited hyperparameter search.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import logging
from src.config import DEFAULT_CONFIG, get_device
from src.data import load_data, create_rolling_windows
from src.train import train_model, calculate_top5_metrics
import torch
import time

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_temporal_validation_split(train_X: np.ndarray, train_y: np.ndarray, 
                                   val_split: float = 0.2):
    """Create temporal validation split using the most recent data from training window."""
    n_samples = len(train_X)
    split_idx = int(n_samples * (1 - val_split))
    
    # Use LAST 20% of training window for validation (most recent data before prediction)
    train_X_split = train_X[:split_idx]
    train_y_split = train_y[:split_idx]
    val_X = train_X[split_idx:]
    val_y = train_y[split_idx:]
    
    return train_X_split, train_y_split, val_X, val_y

def evaluate_config_fixed(config, train_data, device):
    """Evaluate a single config with fixed temporal validation."""
    all_returns = []
    all_hit_rates = []
    
    for i, (train_X, train_y, target_X, target_y, date) in enumerate(train_data):
        try:
            # Create temporal validation split (NO LEAKAGE)
            train_X_split, train_y_split, val_X, val_y = create_temporal_validation_split(
                train_X, train_y, val_split=0.2
            )
            
            # Train model with proper temporal split
            full_config = config.copy()
            full_config.update({
                'n_epochs': 50,  # Reduced for speed
                'early_stopping_patience': 8,
                'val_split': 0.0,
                'random_seed': 42
            })
            
            model, history = train_model(train_X_split, train_y_split, val_X, val_y, full_config)
            
            # Evaluate
            model.eval()
            with torch.no_grad():
                val_X_tensor = torch.FloatTensor(val_X).to(device)
                val_y_tensor = torch.FloatTensor(val_y).to(device)
                predictions = model(val_X_tensor)
                
                metrics = calculate_top5_metrics(predictions, val_y_tensor)
                all_returns.append(metrics['avg_top5_return'])
                all_hit_rates.append(metrics['hit_rate'])
                
        except Exception as e:
            logger.warning(f"Failed on month {i+1}: {e}")
            all_returns.append(-0.1)
            all_hit_rates.append(0.0)
    
    return {
        'avg_return': np.mean(all_returns),
        'avg_hit_rate': np.mean(all_hit_rates),
        'return_std': np.std(all_returns)
    }

def run_quick_fix_test():
    """Run quick test of fixed system."""
    logger.info("Starting quick fix test")
    
    # Load data
    forecast_df, actual_df, factor_names, dates = load_data()
    logger.info(f"Loaded data: {len(forecast_df)} months, {len(factor_names)} factors")
    
    # Create rolling windows (use first 10 for quick test)
    rolling_windows = create_rolling_windows(forecast_df, actual_df, window_size=60)
    test_data = rolling_windows[:10]
    logger.info(f"Using {len(test_data)} windows for quick test")
    
    device = get_device()
    logger.info(f"Using device: {device}")
    
    # Test configurations
    test_configs = [
        # Original best (with leakage)
        {'hidden_sizes': [512, 256], 'learning_rate': 0.001, 'dropout_rate': 0.4, 
         'batch_size': 32, 'weight_decay': 1e-5},
        
        # Alternative configs
        {'hidden_sizes': [256], 'learning_rate': 0.001, 'dropout_rate': 0.2, 
         'batch_size': 32, 'weight_decay': 1e-5},
         
        {'hidden_sizes': [512, 256], 'learning_rate': 0.0001, 'dropout_rate': 0.2, 
         'batch_size': 16, 'weight_decay': 1e-5},
    ]
    
    results = []
    for i, config in enumerate(test_configs):
        logger.info(f"Testing config {i+1}/3: {config}")
        start_time = time.time()
        
        metrics = evaluate_config_fixed(config, test_data, device)
        
        elapsed = time.time() - start_time
        logger.info(f"Config {i+1} completed in {elapsed:.1f}s: "
                   f"Return={metrics['avg_return']:.4f}, Hit Rate={metrics['avg_hit_rate']:.3f}")
        
        result = config.copy()
        result.update(metrics)
        results.append(result)
    
    # Find best config
    best_config = max(results, key=lambda x: x['avg_return'])
    logger.info("\n" + "="*60)
    logger.info("QUICK FIX TEST RESULTS")
    logger.info("="*60)
    
    for i, result in enumerate(results):
        logger.info(f"Config {i+1}: Return={result['avg_return']:.4f}, "
                   f"Hit Rate={result['avg_hit_rate']:.3f}, Std={result['return_std']:.4f}")
    
    logger.info(f"\nBest config: {best_config}")
    
    # Save best config
    with open('outputs/best_config_quick_fix.py', 'w') as f:
        f.write("# Best config from quick fix test (no temporal leakage)\n")
        config_dict = {k: v for k, v in best_config.items() 
                      if k in ['hidden_sizes', 'learning_rate', 'dropout_rate', 'batch_size', 'weight_decay']}
        f.write(f"BEST_CONFIG_FIXED = {config_dict}\n")
    
    logger.info("Quick fix test completed!")
    return best_config

if __name__ == "__main__":
    run_quick_fix_test()