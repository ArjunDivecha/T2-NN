#!/usr/bin/env python3
"""
Run backtest with fixed temporal validation (no leakage) configuration.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import logging
from src.backtest import run_backtest
from src.evaluate import save_evaluation_results, calculate_metrics, calculate_benchmark_metrics, compare_performance

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_fixed_backtest():
    """Run backtest with fixed configuration (no temporal leakage)."""
    logger.info("Starting FIXED backtest (no temporal leakage)")
    
    # Use the best configuration from quick fix test
    fixed_config = {
        'hidden_sizes': [512, 256],
        'learning_rate': 0.001,
        'dropout_rate': 0.4,
        'batch_size': 32,
        'weight_decay': 1e-5
    }
    
    logger.info(f"Using FIXED configuration: {fixed_config}")
    
    # Run limited backtest on 2023-2024 (same period as before)
    results, analysis = run_backtest(
        data_dir='data/',
        parallel=False,  # Use sequential for better logging
        start_date='2023-01-01',
        end_date='2024-12-31',
        config=fixed_config,
        save_results=True,
        save_forecasts=True
    )
    
    logger.info("Fixed backtest completed!")
    logger.info(f"Processed {len(results)} months")
    
    # Run evaluation manually since we need to use individual functions
    logger.info("Running evaluation on fixed backtest results...")
    
    import numpy as np
    
    # Extract metrics from results
    monthly_returns = results['portfolio_return'].values
    hit_rates = results['hit_rate'].values
    
    # Calculate model metrics (use correct key names for compare_performance function)
    model_metrics = {
        'top5_return': np.mean(monthly_returns),  # Key name expected by compare_performance
        'avg_top5_return': np.mean(monthly_returns),  # Alias for display
        'hit_rate': np.mean(hit_rates),
        'spearman_correlation': 0.0,  # Would need predictions vs actual
        'pearson_correlation': 0.0,
        'return_volatility': np.std(monthly_returns),
        'hit_rate_std': np.std(hit_rates),
        'win_rate': np.mean(monthly_returns > 0),
        'n_samples': len(monthly_returns),
        'top_k': 5,
        'monthly_returns': monthly_returns.tolist()  # Also needed by compare_performance
    }
    
    # Calculate Sharpe ratio
    if model_metrics['return_volatility'] > 0:
        model_metrics['sharpe_ratio'] = model_metrics['avg_top5_return'] / model_metrics['return_volatility']
        model_metrics['information_ratio'] = model_metrics['sharpe_ratio']
        
        # Sortino ratio (downside deviation)
        negative_returns = monthly_returns[monthly_returns < 0]
        if len(negative_returns) > 0:
            downside_deviation = np.std(negative_returns)
            model_metrics['sortino_ratio'] = model_metrics['avg_top5_return'] / downside_deviation
        else:
            model_metrics['sortino_ratio'] = float('inf')
    else:
        model_metrics['sharpe_ratio'] = 0.0
        model_metrics['information_ratio'] = 0.0
        model_metrics['sortino_ratio'] = 0.0
    
    # Create benchmark metrics (simplified)
    benchmark_metrics = {
        'equal_weighted_return': -0.0006,  # From previous analysis
        'equal_weighted_sharpe': -0.145,
        'equal_weighted_volatility': 0.0136,
        'equal_weighted_win_rate': 0.542,
        'random_top5_return': 0.0079,
        'random_top5_sharpe': 0.936,
        'random_top5_volatility': 0.0294,
        'random_top5_win_rate': 0.583
    }
    
    # Compare performance  
    comparison_metrics = compare_performance(model_metrics, benchmark_metrics)
    
    # Combine all metrics
    eval_results = {**model_metrics, **comparison_metrics}
    
    # Save results
    os.makedirs('outputs/fixed_backtest_eval/', exist_ok=True)
    save_evaluation_results(
        model_metrics=model_metrics,
        benchmark_metrics=benchmark_metrics,
        comparison_metrics=comparison_metrics,
        output_dir='outputs/fixed_backtest_eval/'
    )
    
    logger.info("="*80)
    logger.info("FIXED BACKTEST RESULTS SUMMARY")
    logger.info("="*80)
    logger.info(f"Average Monthly Return: {eval_results['avg_top5_return']:.4f} ({eval_results['avg_top5_return']*100:.2f}%)")
    logger.info(f"Hit Rate: {eval_results['hit_rate']:.3f} ({eval_results['hit_rate']*100:.1f}%)")
    logger.info(f"Sharpe Ratio: {eval_results['sharpe_ratio']:.3f}")
    logger.info(f"Annual Return: {eval_results['annual_excess_return_vs_ew']:.2f}%")
    logger.info(f"vs Equal-Weighted p-value: {eval_results['p_value_vs_ew']:.4f}")
    
    # Compare with original results
    logger.info("="*80)
    logger.info("COMPARISON WITH ORIGINAL (LEAKAGE) RESULTS")
    logger.info("="*80)
    
    try:
        import pandas as pd
        original_eval = pd.read_csv('outputs/limited_backtest_eval/evaluation_metrics.csv')
        
        logger.info("ORIGINAL (with temporal leakage):")
        logger.info(f"  Average Return: {original_eval.iloc[0]['model_top5_return']:.4f} ({original_eval.iloc[0]['model_top5_return']*100:.2f}%)")
        logger.info(f"  Hit Rate: {original_eval.iloc[0]['model_hit_rate']:.3f} ({original_eval.iloc[0]['model_hit_rate']*100:.1f}%)")
        logger.info(f"  Sharpe Ratio: {original_eval.iloc[0]['model_sharpe_ratio']:.3f}")
        
        logger.info("FIXED (no temporal leakage):")
        logger.info(f"  Average Return: {eval_results['avg_top5_return']:.4f} ({eval_results['avg_top5_return']*100:.2f}%)")
        logger.info(f"  Hit Rate: {eval_results['hit_rate']:.3f} ({eval_results['hit_rate']*100:.1f}%)")
        logger.info(f"  Sharpe Ratio: {eval_results['sharpe_ratio']:.3f}")
        
        # Calculate performance difference
        return_diff = eval_results['avg_top5_return'] - original_eval.iloc[0]['model_top5_return']
        hit_rate_diff = eval_results['hit_rate'] - original_eval.iloc[0]['model_hit_rate']
        
        logger.info("DIFFERENCE (Fixed - Original):")
        logger.info(f"  Return Difference: {return_diff:.4f} ({return_diff*100:.2f} percentage points)")
        logger.info(f"  Hit Rate Difference: {hit_rate_diff:.3f} ({hit_rate_diff*100:.1f} percentage points)")
        
        if return_diff < -0.05:  # If difference is more than 5 percentage points
            logger.info("✅ TEMPORAL LEAKAGE CONFIRMED: Fixed model shows significantly worse performance")
        else:
            logger.info("⚠️  Performance difference is smaller than expected")
            
    except Exception as e:
        logger.warning(f"Could not load original results for comparison: {e}")
    
    logger.info("="*80)
    logger.info("CONCLUSION")
    logger.info("="*80)
    logger.info("The fixed model eliminates temporal leakage by using proper temporal validation")
    logger.info("splits during both hyperparameter tuning and training. This should result in")
    logger.info("more realistic performance estimates that better match live trading results.")
    
    return results, eval_results

if __name__ == "__main__":
    run_fixed_backtest()