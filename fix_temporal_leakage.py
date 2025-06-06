#!/usr/bin/env python3
"""
Fix temporal leakage in hyperparameter tuning.

The current implementation uses the target month's actual returns as validation data,
creating forward-looking bias. This script fixes the issue by implementing proper
temporal validation splits.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional
from src.config import DEFAULT_CONFIG
from src.data import load_data, create_rolling_windows
from src.train import train_model, calculate_top5_metrics
from src.model import SimpleNN
import torch
import time
import random

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_temporal_validation_split(train_X: np.ndarray, train_y: np.ndarray, 
                                   val_split: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create temporal validation split using the most recent data from training window.
    
    Args:
        train_X: Training features
        train_y: Training targets  
        val_split: Fraction to use for validation (default 0.2 = 20%)
        
    Returns:
        train_X_split, train_y_split, val_X, val_y
    """
    n_samples = len(train_X)
    split_idx = int(n_samples * (1 - val_split))
    
    # Use LAST 20% of training window for validation (most recent data)
    train_X_split = train_X[:split_idx]
    train_y_split = train_y[:split_idx]
    val_X = train_X[split_idx:]
    val_y = train_y[split_idx:]
    
    logger.debug(f"Temporal split: {len(train_X_split)} train, {len(val_X)} validation samples")
    
    return train_X_split, train_y_split, val_X, val_y

def evaluate_hyperparameters_fixed(params: Dict, train_data: List, device: torch.device) -> Dict:
    """
    Fixed version of hyperparameter evaluation without temporal leakage.
    
    Args:
        params: Hyperparameter configuration
        train_data: List of (train_X, train_y, target_X, target_y, date) tuples
        device: PyTorch device
        
    Returns:
        Dictionary with evaluation metrics
    """
    logger.info(f"Evaluating hyperparameters (FIXED): {params}")
    
    # Track metrics across all training windows
    all_returns = []
    all_hit_rates = []
    all_train_times = []
    all_epochs = []
    
    for i, (train_X, train_y, target_X, target_y, date) in enumerate(train_data):
        try:
            # Create temporal validation split from training data (NO LEAKAGE)
            train_X_split, train_y_split, val_X, val_y = create_temporal_validation_split(
                train_X, train_y, val_split=0.2
            )
            
            # Create training configuration
            config = params.copy()
            config.update({
                'n_epochs': 100,
                'early_stopping_patience': 10,
                'val_split': 0.0,  # Use our pre-split validation data
                'random_seed': 42
            })
            
            start_time = time.time()
            
            # Train model with proper temporal split
            model, history = train_model(train_X_split, train_y_split, val_X, val_y, config)
            
            train_time = time.time() - start_time
            
            # Evaluate on validation set (still within training window, no leakage)
            model.eval()
            with torch.no_grad():
                val_X_tensor = torch.FloatTensor(val_X).to(device)
                val_y_tensor = torch.FloatTensor(val_y).to(device)
                predictions = model(val_X_tensor)
                
                metrics = calculate_top5_metrics(predictions, val_y_tensor)
                
                all_returns.append(metrics['avg_top5_return'])
                all_hit_rates.append(metrics['hit_rate'])
                all_train_times.append(train_time)
                all_epochs.append(history['final_epoch'])
            
            logger.debug(f"Month {i+1}/{len(train_data)} - Return: {metrics['avg_top5_return']:.4f}, "
                        f"Hit Rate: {metrics['hit_rate']:.3f}, Time: {train_time:.2f}s")
                        
        except Exception as e:
            logger.warning(f"Failed to train on month {i+1}: {e}")
            # Use poor metrics for failed training
            all_returns.append(-0.1)  # Very poor return
            all_hit_rates.append(0.0)  # No hits
            all_train_times.append(0.0)
            all_epochs.append(0)
    
    # Calculate aggregate metrics
    valid_returns = [r for r in all_returns if r is not None and not np.isnan(r)]
    valid_hit_rates = [h for h in all_hit_rates if h is not None and not np.isnan(h)]
    
    if not valid_returns:
        logger.warning("No valid returns found")
        return {
            'avg_top5_return': -0.1,
            'avg_hit_rate': 0.0,
            'avg_train_time': np.mean([t for t in all_train_times if t > 0]) if all_train_times else 0.0,
            'avg_epochs': np.mean([e for e in all_epochs if e > 0]) if all_epochs else 0.0,
            'return_std': 0.0,
            'hit_rate_std': 0.0,
            'success_rate': 0.0
        }
    
    # Calculate success rate (fraction of successful training runs)
    success_rate = len(valid_returns) / len(all_returns)
    
    results = {
        'avg_top5_return': np.mean(valid_returns),
        'avg_hit_rate': np.mean(valid_hit_rates),
        'avg_train_time': np.mean([t for t in all_train_times if t > 0]),
        'avg_epochs': np.mean([e for e in all_epochs if e > 0]),
        'return_std': np.std(valid_returns),
        'hit_rate_std': np.std(valid_hit_rates),
        'success_rate': success_rate
    }
    
    logger.info(f"Results: Avg Return: {results['avg_top5_return']:.4f}, "
               f"Avg Hit Rate: {results['avg_hit_rate']:.3f}, "
               f"Success Rate: {results['success_rate']:.3f}")
    
    return results

def run_fixed_hyperparameter_comparison():
    """
    Compare hyperparameter tuning with and without temporal leakage fix.
    """
    logger.info("Starting fixed hyperparameter tuning comparison")
    
    # Load data
    forecast_df, actual_df, factor_names, dates = load_data()
    logger.info(f"Loaded data: {len(forecast_df)} months, {len(factor_names)} factors")
    
    # Create limited training data (first 50 months for faster testing)
    rolling_windows = create_rolling_windows(forecast_df, actual_df, window_size=60)
    
    # Use first 10 windows for comparison
    train_data = rolling_windows[:10]
    logger.info(f"Using {len(train_data)} training windows for comparison")
    
    # Test configuration (single config for comparison)
    test_config = {
        'hidden_sizes': [512, 256],
        'learning_rate': 0.001,
        'dropout_rate': 0.4,
        'batch_size': 32,
        'weight_decay': 1e-5
    }
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Run fixed evaluation
    logger.info("=" * 60)
    logger.info("RUNNING FIXED HYPERPARAMETER EVALUATION (NO TEMPORAL LEAKAGE)")
    logger.info("=" * 60)
    
    fixed_results = evaluate_hyperparameters_fixed(test_config, train_data, device)
    
    logger.info("=" * 60)
    logger.info("COMPARISON RESULTS")
    logger.info("=" * 60)
    logger.info(f"Fixed Method (No Leakage):")
    logger.info(f"  Average Return: {fixed_results['avg_top5_return']:.4f} ({fixed_results['avg_top5_return']*100:.2f}%)")
    logger.info(f"  Average Hit Rate: {fixed_results['avg_hit_rate']:.3f} ({fixed_results['avg_hit_rate']*100:.1f}%)")
    logger.info(f"  Return Std: {fixed_results['return_std']:.4f}")
    logger.info(f"  Success Rate: {fixed_results['success_rate']:.3f}")
    
    # Load previous results for comparison
    try:
        import pandas as pd
        hyperparam_df = pd.read_csv('outputs/hyperparam_results.csv')
        best_row = hyperparam_df.iloc[0]  # First row is best
        
        logger.info(f"Original Method (With Leakage):")
        logger.info(f"  Average Return: {best_row['avg_top5_return']:.4f} ({best_row['avg_top5_return']*100:.2f}%)")
        logger.info(f"  Average Hit Rate: {best_row['avg_hit_rate']:.3f} ({best_row['avg_hit_rate']*100:.1f}%)")
        logger.info(f"  Return Std: {best_row['return_std']:.4f}")
        logger.info(f"  Success Rate: {best_row['success_rate']:.3f}")
        
        logger.info("=" * 60)
        logger.info("ANALYSIS:")
        logger.info("The fixed method should show significantly lower performance,")
        logger.info("confirming that the original high performance was due to temporal leakage.")
        
    except Exception as e:
        logger.warning(f"Could not load previous results: {e}")

if __name__ == "__main__":
    run_fixed_hyperparameter_comparison()