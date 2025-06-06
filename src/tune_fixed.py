#!/usr/bin/env python3
"""
Fixed hyperparameter tuning without temporal leakage.

This module implements proper temporal validation for hyperparameter optimization,
eliminating the forward-looking bias present in the original implementation.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import logging
import time
import torch
import itertools
from typing import Dict, List, Tuple, Optional
from .config import HYPERPARAM_GRID, OUTPUT_DIR, DEFAULT_CONFIG, get_device
from .data import load_data, create_rolling_windows
from .train import train_model, calculate_top5_metrics
from .model import SimpleNN

logger = logging.getLogger(__name__)

def create_temporal_validation_split(train_X: np.ndarray, train_y: np.ndarray, 
                                   val_split: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create temporal validation split using the most recent data from training window.
    
    Args:
        train_X: Training features (chronologically ordered)
        train_y: Training targets
        val_split: Fraction to use for validation (default 0.2 = 20%)
        
    Returns:
        train_X_split, train_y_split, val_X, val_y
    """
    n_samples = len(train_X)
    split_idx = int(n_samples * (1 - val_split))
    
    # Use LAST 20% of training window for validation (most recent data before prediction)
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

def run_fixed_hyperparameter_tuning(n_random_months: int = 50, save_results: bool = True) -> pd.DataFrame:
    """
    Run full hyperparameter tuning with fixed temporal validation.
    
    Args:
        n_random_months: Number of random months to use for validation
        save_results: Whether to save results to CSV
        
    Returns:
        DataFrame with hyperparameter results sorted by performance
    """
    logger.info("Starting FIXED hyperparameter tuning (no temporal leakage)")
    
    # Load data
    forecast_df, actual_df, factor_names, dates = load_data()
    logger.info(f"Loaded data: {len(forecast_df)} months, {len(factor_names)} factors")
    
    # Create rolling windows
    rolling_windows = create_rolling_windows(forecast_df, actual_df, window_size=60)
    logger.info(f"Created {len(rolling_windows)} rolling windows")
    
    # Select random months for hyperparameter validation
    np.random.seed(42)  # For reproducibility
    selected_indices = np.random.choice(len(rolling_windows), size=min(n_random_months, len(rolling_windows)), replace=False)
    train_data = [rolling_windows[i] for i in sorted(selected_indices)]
    
    logger.info(f"Selected {len(train_data)} random months for hyperparameter tuning")
    logger.info(f"Date range: {train_data[0][4]} to {train_data[-1][4]}")
    
    # Generate all hyperparameter combinations
    param_names = list(HYPERPARAM_GRID.keys())
    param_values = list(HYPERPARAM_GRID.values())
    param_combinations = list(itertools.product(*param_values))
    
    logger.info(f"Testing {len(param_combinations)} hyperparameter combinations")
    
    # Get device
    device = get_device()
    logger.info(f"Using device: {device}")
    
    # Test all combinations
    results = []
    
    for i, param_combo in enumerate(param_combinations):
        # Create parameter dictionary
        params = dict(zip(param_names, param_combo))
        
        logger.info(f"Testing combination {i+1}/{len(param_combinations)}: {params}")
        
        # Evaluate this combination
        metrics = evaluate_hyperparameters_fixed(params, train_data, device)
        
        # Store results
        result_row = params.copy()
        result_row.update(metrics)
        results.append(result_row)
        
        logger.info(f"Combination {i+1} completed: Return={metrics['avg_top5_return']:.4f}, "
                   f"Hit Rate={metrics['avg_hit_rate']:.3f}")
    
    # Convert to DataFrame and sort by performance
    results_df = pd.DataFrame(results)
    
    # Sort by average return (descending), then by hit rate (descending)
    results_df = results_df.sort_values(['avg_top5_return', 'avg_hit_rate'], ascending=[False, False])
    results_df = results_df.reset_index(drop=True)
    
    logger.info("Hyperparameter tuning completed!")
    logger.info(f"Best configuration: {dict(results_df.iloc[0][param_names])}")
    logger.info(f"Best performance: Return={results_df.iloc[0]['avg_top5_return']:.4f}, "
               f"Hit Rate={results_df.iloc[0]['avg_hit_rate']:.3f}")
    
    if save_results:
        # Save results
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_file = f"{OUTPUT_DIR}hyperparam_results_fixed.csv"
        results_df.to_csv(output_file, index=False)
        logger.info(f"Results saved to {output_file}")
        
        # Update default configuration with best parameters
        best_params = dict(results_df.iloc[0][param_names])
        logger.info(f"Best hyperparameters: {best_params}")
        
        # Save best config for future use
        best_config_file = f"{OUTPUT_DIR}best_config_fixed.py"
        with open(best_config_file, 'w') as f:
            f.write("# Best hyperparameters from fixed tuning (no temporal leakage)\n")
            f.write(f"BEST_CONFIG_FIXED = {best_params}\n")
        logger.info(f"Best configuration saved to {best_config_file}")
    
    return results_df

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Run hyperparameter tuning
    results_df = run_fixed_hyperparameter_tuning(n_random_months=50, save_results=True)
    
    print("\n" + "="*80)
    print("FIXED HYPERPARAMETER TUNING RESULTS")
    print("="*80)
    print(f"Top 5 configurations:")
    print(results_df.head())
    
    print(f"\nBest configuration:")
    best_row = results_df.iloc[0]
    print(f"Hidden sizes: {best_row['hidden_sizes']}")
    print(f"Learning rate: {best_row['learning_rate']}")
    print(f"Dropout rate: {best_row['dropout_rate']}")
    print(f"Batch size: {best_row['batch_size']}")
    print(f"Weight decay: {best_row['weight_decay']}")
    print(f"Average return: {best_row['avg_top5_return']:.4f} ({best_row['avg_top5_return']*100:.2f}%)")
    print(f"Average hit rate: {best_row['avg_hit_rate']:.3f} ({best_row['avg_hit_rate']*100:.1f}%)")
    print(f"Return volatility: {best_row['return_std']:.4f}")
    print(f"Success rate: {best_row['success_rate']:.3f}")