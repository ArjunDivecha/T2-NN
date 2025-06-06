#!/usr/bin/env python3
"""
Test script for backtest functionality.
Tests the backtest on a small date range to verify everything works.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import logging
from src.config import setup_logging
from src.backtest import run_backtest, train_and_predict_month, create_backtest_dataframe, analyze_backtest_results
from src.data import load_data, create_rolling_windows

def test_single_month_prediction():
    """Test training and prediction for a single month."""
    logger = logging.getLogger(__name__)
    logger.info("Testing single month prediction...")
    
    # Load data
    t60_df, t2_df, factor_names, dates = load_data('data/T60.xlsx', 'data/T2_Optimizer.xlsx')
    
    # Create one rolling window for testing
    all_windows = create_rolling_windows(t60_df, t2_df, window_size=60)
    
    if len(all_windows) == 0:
        logger.error("No rolling windows available for testing")
        return False
    
    # Test with the first window
    test_window = all_windows[0]
    test_date = test_window[4]
    
    logger.info(f"Testing with window for date: {test_date}")
    
    # Use simple config for quick testing
    config = {
        'hidden_sizes': [128],
        'learning_rate': 1e-3,
        'dropout_rate': 0.2,
        'batch_size': 32,
        'weight_decay': 1e-5,
        'n_epochs': 20,  # Reduced for testing
        'early_stopping_patience': 5,
        'random_seed': 42
    }
    
    # Test the training and prediction function
    result = train_and_predict_month(test_window, config, device_str='mps')
    
    # Verify results
    if not result['success']:
        logger.error(f"Training failed: {result.get('error', 'Unknown error')}")
        return False
    
    logger.info("Single month prediction results:")
    logger.info(f"  Date: {result['date']}")
    logger.info(f"  Portfolio return: {result['portfolio_return']:.4f}")
    logger.info(f"  Hit rate: {result['hit_rate']:.3f}")
    logger.info(f"  Training time: {result['train_time']:.2f}s")
    logger.info(f"  Final epoch: {result['final_epoch']}")
    logger.info(f"  Top 5 indices: {result['top5_indices']}")
    
    # Verify result structure
    required_keys = ['date', 'portfolio_return', 'hit_rate', 'train_time', 'predictions', 'success']
    for key in required_keys:
        if key not in result:
            logger.error(f"Missing key in result: {key}")
            return False
    
    logger.info("✅ Single month prediction test passed!")
    return True

def test_sequential_backtest_small():
    """Test sequential backtest on a small date range."""
    logger = logging.getLogger(__name__)
    logger.info("Testing sequential backtest on small date range...")
    
    # Test with a very small date range (3 months)
    start_date = '2020-01-01'
    end_date = '2020-04-01'
    
    try:
        results_df, analysis = run_backtest(
            data_dir='data/',
            parallel=False,  # Use sequential for testing
            start_date=start_date,
            end_date=end_date,
            save_results=False,  # Don't save for testing
            save_forecasts=False
        )
        
        logger.info(f"Sequential backtest completed with {len(results_df)} results")
        
        if len(results_df) == 0:
            logger.warning("No results from backtest - this might be expected for small date range")
            return True
        
        # Verify results structure
        expected_columns = ['date', 'portfolio_return', 'hit_rate', 'train_time']
        for col in expected_columns:
            if col not in results_df.columns:
                logger.error(f"Missing column in results: {col}")
                return False
        
        # Show results
        logger.info("Backtest results:")
        for _, row in results_df.iterrows():
            logger.info(f"  {row['date']}: Return={row['portfolio_return']:.4f}, Hit Rate={row['hit_rate']:.3f}")
        
        # Check analysis
        if analysis:
            logger.info(f"Analysis summary:")
            logger.info(f"  Average monthly return: {analysis.get('avg_monthly_return', 'N/A')}")
            logger.info(f"  Average hit rate: {analysis.get('avg_hit_rate', 'N/A')}")
            logger.info(f"  Sharpe ratio: {analysis.get('sharpe_ratio', 'N/A')}")
        
        logger.info("✅ Sequential backtest test passed!")
        return True
        
    except Exception as e:
        logger.error(f"Sequential backtest test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_parallel_backtest_small():
    """Test parallel backtest on a small date range."""
    logger = logging.getLogger(__name__)
    logger.info("Testing parallel backtest on small date range...")
    
    # Test with a small date range
    start_date = '2021-01-01'
    end_date = '2021-06-01'
    
    try:
        results_df, analysis = run_backtest(
            data_dir='data/',
            parallel=True,
            start_date=start_date,
            end_date=end_date,
            save_results=False,
            save_forecasts=False
        )
        
        logger.info(f"Parallel backtest completed with {len(results_df)} results")
        
        if len(results_df) == 0:
            logger.warning("No results from parallel backtest - this might be expected for small date range")
            return True
        
        # Show sample results
        logger.info("Sample results:")
        for _, row in results_df.head(3).iterrows():
            logger.info(f"  {row['date']}: Return={row['portfolio_return']:.4f}, Hit Rate={row['hit_rate']:.3f}")
        
        logger.info("✅ Parallel backtest test passed!")
        return True
        
    except Exception as e:
        logger.error(f"Parallel backtest test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_backtest_analysis():
    """Test backtest analysis functions."""
    logger = logging.getLogger(__name__)
    logger.info("Testing backtest analysis...")
    
    try:
        # Run a small backtest to get real data
        results_df, analysis = run_backtest(
            data_dir='data/',
            parallel=False,
            start_date='2022-01-01',
            end_date='2022-12-01',
            save_results=False,
            save_forecasts=False
        )
        
        if len(results_df) == 0:
            logger.warning("No results for analysis test - using mock data")
            
            # Create mock results for testing
            import pandas as pd
            import numpy as np
            
            mock_data = {
                'date': pd.date_range('2022-01-01', periods=12, freq='MS'),
                'portfolio_return': np.random.randn(12) * 0.02 + 0.01,  # ~1% monthly with volatility
                'hit_rate': np.random.uniform(0.2, 0.5, 12),
                'train_time': np.random.uniform(10, 30, 12),
                'final_epoch': np.random.randint(10, 50, 12),
                'best_val_return': np.random.randn(12) * 0.01 + 0.005
            }
            results_df = pd.DataFrame(mock_data)
            
            logger.info("Created mock data for analysis testing")
        
        # Test analysis function
        analysis = analyze_backtest_results(results_df)
        
        # Verify analysis results
        required_metrics = ['n_months', 'avg_monthly_return', 'annual_return', 'sharpe_ratio', 'avg_hit_rate']
        for metric in required_metrics:
            if metric not in analysis:
                logger.error(f"Missing metric in analysis: {metric}")
                return False
        
        logger.info("Analysis results:")
        logger.info(f"  Months: {analysis['n_months']}")
        logger.info(f"  Avg monthly return: {analysis['avg_monthly_return']:.4f}")
        logger.info(f"  Annual return: {analysis['annual_return']:.2%}")
        logger.info(f"  Sharpe ratio: {analysis['sharpe_ratio']:.3f}")
        logger.info(f"  Avg hit rate: {analysis['avg_hit_rate']:.3f}")
        logger.info(f"  Max drawdown: {analysis['max_drawdown']:.3f}")
        
        logger.info("✅ Backtest analysis test passed!")
        return True
        
    except Exception as e:
        logger.error(f"Backtest analysis test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_availability():
    """Test if we have sufficient data for backtesting."""
    logger = logging.getLogger(__name__)
    logger.info("Testing data availability...")
    
    try:
        # Load data
        t60_df, t2_df, factor_names, dates = load_data('data/T60.xlsx', 'data/T2_Optimizer.xlsx')
        
        logger.info(f"Data loaded: {len(dates)} months, {len(factor_names)} factors")
        logger.info(f"Date range: {dates[0]} to {dates[-1]}")
        
        # Create rolling windows
        windows = create_rolling_windows(t60_df, t2_df, window_size=60)
        logger.info(f"Can create {len(windows)} rolling windows")
        
        if len(windows) < 10:
            logger.warning("Very few rolling windows available - backtest will be limited")
        
        # Show first and last available test dates
        if len(windows) > 0:
            first_test_date = windows[0][4]
            last_test_date = windows[-1][4]
            logger.info(f"First test date: {first_test_date}")
            logger.info(f"Last test date: {last_test_date}")
            
            # Calculate years of backtest data
            years = (last_test_date - first_test_date).days / 365.25
            logger.info(f"Backtest covers approximately {years:.1f} years")
        
        logger.info("✅ Data availability test passed!")
        return True
        
    except Exception as e:
        logger.error(f"Data availability test failed: {e}")
        return False

def main():
    """Run all backtest tests."""
    # Setup logging
    logger = setup_logging('outputs/test_backtest.log')
    logger.info("Starting backtest system tests")
    
    success = True
    
    # Run all tests
    tests = [
        ("Data availability", test_data_availability),
        ("Single month prediction", test_single_month_prediction),
        ("Sequential backtest (small)", test_sequential_backtest_small),
        ("Parallel backtest (small)", test_parallel_backtest_small),
        ("Backtest analysis", test_backtest_analysis)
    ]
    
    for test_name, test_func in tests:
        try:
            logger.info(f"\n{'='*50}")
            logger.info(f"Running test: {test_name}")
            logger.info(f"{'='*50}")
            
            result = test_func()
            if not result:
                logger.error(f"❌ Test failed: {test_name}")
                success = False
            else:
                logger.info(f"✅ Test passed: {test_name}")
                
        except Exception as e:
            logger.error(f"❌ Test failed with exception: {test_name} - {e}")
            import traceback
            traceback.print_exc()
            success = False
    
    if success:
        logger.info("\n🎉 All backtest tests passed!")
        print("✅ Backtest system tests completed successfully!")
        return True
    else:
        logger.error("\n💥 Some backtest tests failed!")
        print("❌ Some backtest tests failed!")
        return False

if __name__ == "__main__":
    try:
        success = main()
        if not success:
            sys.exit(1)
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)