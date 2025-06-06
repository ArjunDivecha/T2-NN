#!/usr/bin/env python3
"""Test script for data loading functionality"""

import sys
import os
sys.path.append('src')

from src.config import setup_logging
from src.data import load_data, create_rolling_windows, select_random_months, get_data_statistics

def main():
    # Setup logging
    logger = setup_logging()
    logger.info("Testing data loading functionality...")
    
    try:
        # Load data
        t60_df, t2_df, factor_names, dates = load_data()
        
        # Print basic info
        print(f"\n📊 Data loaded successfully!")
        print(f"Shape: {t60_df.shape}")
        print(f"Factors: {len(factor_names)}")
        print(f"Date range: {dates[0]} to {dates[-1]}")
        
        # Get statistics
        stats = get_data_statistics(t60_df, t2_df)
        print(f"\n📈 Data Statistics:")
        print(f"T60 (forecasts): mean={stats['t60_stats']['mean']:.4f}, std={stats['t60_stats']['std']:.4f}")
        print(f"T2 (actuals): mean={stats['t2_stats']['mean']:.4f}, std={stats['t2_stats']['std']:.4f}")
        
        # Create rolling windows
        windows = create_rolling_windows(t60_df, t2_df, window_size=60)
        print(f"\n🪟 Created {len(windows)} rolling windows")
        
        # Test first window
        train_X, train_y, test_X, test_y, test_date = windows[0]
        print(f"First window - Train: {train_X.shape}, Test: {test_X.shape}, Date: {test_date}")
        
        # Select random months for tuning
        random_windows = select_random_months(windows, n_months=30)
        print(f"\n🎲 Selected {len(random_windows)} random windows for hyperparameter tuning")
        
        print("\n✅ All data tests passed!")
        
    except Exception as e:
        logger.error(f"Data test failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()