#!/usr/bin/env python3
"""
Inspect the training data splits to understand what's happening.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import logging
from src.data import load_data, create_rolling_windows

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def inspect_training_window(window_index=0):
    """
    Inspect a specific training window to see exactly what data is used.
    
    Args:
        window_index: Which rolling window to inspect (0 = first window)
    """
    logger.info(f"Inspecting training window {window_index}")
    
    # Load data
    forecast_df, actual_df, factor_names, dates = load_data()
    logger.info(f"Total data: {len(forecast_df)} months from {dates[0]} to {dates[-1]}")
    
    # Create rolling windows
    rolling_windows = create_rolling_windows(forecast_df, actual_df, window_size=60)
    logger.info(f"Created {len(rolling_windows)} rolling windows")
    
    if window_index >= len(rolling_windows):
        logger.error(f"Window index {window_index} is out of range. Max index: {len(rolling_windows)-1}")
        return
    
    # Get the specific window
    train_X, train_y, target_X, target_y, target_date = rolling_windows[window_index]
    
    logger.info("="*80)
    logger.info(f"WINDOW {window_index} DATA BREAKDOWN")
    logger.info("="*80)
    
    # Show target date
    logger.info(f"TARGET MONTH (what we're predicting): {target_date}")
    
    # Reconstruct the date range for training data
    target_date_idx = dates.get_loc(target_date)
    train_start_idx = target_date_idx - 60
    train_end_idx = target_date_idx
    
    train_dates = dates[train_start_idx:train_end_idx]
    train_start_date = train_dates[0]
    train_end_date = train_dates[-1]
    
    logger.info(f"TRAINING DATA PERIOD: {train_start_date} to {train_end_date} ({len(train_dates)} months)")
    logger.info(f"Training data shape: X={train_X.shape}, y={train_y.shape}")
    logger.info(f"Target data shape: X={target_X.shape}, y={target_y.shape}")
    
    # Show the overlap
    logger.info("\nDATE ANALYSIS:")
    logger.info(f"  Training ends on: {train_end_date}")
    logger.info(f"  Target month is: {target_date}")
    logger.info(f"  Gap between training and target: {(target_date - train_end_date).days} days")
    
    # Check what's in the training data vs target
    logger.info("\nDATA INSPECTION:")
    logger.info("First 5 training months:")
    for i in range(5):
        logger.info(f"  {train_dates[i]}: forecast avg = {np.mean(train_X[i]):.4f}, actual avg = {np.mean(train_y[i]):.4f}")
    
    logger.info("Last 5 training months:")
    for i in range(-5, 0):
        logger.info(f"  {train_dates[i]}: forecast avg = {np.mean(train_X[i]):.4f}, actual avg = {np.mean(train_y[i]):.4f}")
    
    logger.info(f"Target month:")
    logger.info(f"  {target_date}: forecast avg = {np.mean(target_X):.4f}, actual avg = {np.mean(target_y):.4f}")
    
    # Show current validation split behavior
    logger.info("\n" + "="*80)
    logger.info("CURRENT VALIDATION SPLIT (what I implemented)")
    logger.info("="*80)
    
    # My current temporal split
    val_split = 0.2
    n_samples = len(train_X)
    split_idx = int(n_samples * (1 - val_split))
    
    train_X_split = train_X[:split_idx]
    train_y_split = train_y[:split_idx]
    val_X = train_X[split_idx:]
    val_y = train_y[split_idx:]
    
    val_start_idx = train_start_idx + split_idx
    val_end_idx = train_end_idx
    val_dates = dates[val_start_idx:val_end_idx]
    
    logger.info(f"TRAINING SPLIT: {train_dates[0]} to {train_dates[split_idx-1]} ({len(train_X_split)} months)")
    logger.info(f"VALIDATION SPLIT: {val_dates[0]} to {val_dates[-1]} ({len(val_X)} months)")
    logger.info(f"TARGET: {target_date}")
    
    # Check for temporal leakage
    logger.info("\nTEMPORAL LEAKAGE CHECK:")
    if val_dates[-1] >= target_date:
        logger.error("🚨 TEMPORAL LEAKAGE: Validation data includes or goes beyond target date!")
    else:
        days_gap = (target_date - val_dates[-1]).days
        logger.info(f"✅ No direct leakage: {days_gap} days between last validation and target")
    
    # Show what SHOULD happen for proper validation
    logger.info("\n" + "="*80)
    logger.info("WHAT PROPER VALIDATION SHOULD LOOK LIKE")
    logger.info("="*80)
    logger.info("For hyperparameter tuning, we should use COMPLETELY DIFFERENT months")
    logger.info("that are not part of any backtest window, such as:")
    logger.info("- Training: 2005-2010 (for example)")
    logger.info("- Validation: 2011-2012 (for example)")  
    logger.info("- Backtest: 2013+ (completely separate)")
    logger.info("\nOR use walk-forward validation with proper gaps between periods")

def inspect_multiple_windows():
    """Inspect several windows to see the pattern."""
    logger.info("="*80)
    logger.info("INSPECTING MULTIPLE WINDOWS")
    logger.info("="*80)
    
    # Load data
    forecast_df, actual_df, factor_names, dates = load_data()
    rolling_windows = create_rolling_windows(forecast_df, actual_df, window_size=60)
    
    # Check windows around the backtest period (2023-2024)
    backtest_start = pd.Timestamp('2023-01-01')
    
    for i in range(len(rolling_windows)):
        train_X, train_y, target_X, target_y, target_date = rolling_windows[i]
        
        if target_date >= backtest_start:
            logger.info(f"Window {i}: Target = {target_date}")
            
            # Show training period
            target_date_idx = dates.get_loc(target_date)
            train_start_idx = target_date_idx - 60
            train_end_idx = target_date_idx
            train_dates = dates[train_start_idx:train_end_idx]
            
            logger.info(f"  Training: {train_dates[0]} to {train_dates[-1]}")
            logger.info(f"  Gap: {(target_date - train_dates[-1]).days} days")
            
            if i >= 5:  # Just show first few
                break

if __name__ == "__main__":
    # Inspect the first window in detail
    inspect_training_window(window_index=0)
    
    print("\n" + "="*100 + "\n")
    
    # Inspect windows around backtest period
    inspect_multiple_windows()