import numpy as np
import pandas as pd
import torch
import logging
from typing import Dict, List, Tuple, Optional, Union
from scipy import stats
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt
import os

from .config import OUTPUT_DIR, PLOTS_DIR

logger = logging.getLogger(__name__)

def calculate_metrics(predictions: Union[np.ndarray, torch.Tensor], 
                     actual_returns: Union[np.ndarray, torch.Tensor],
                     top_k: int = 5) -> Dict[str, float]:
    """
    Calculate comprehensive evaluation metrics for factor predictions.
    
    Args:
        predictions: Model predictions of shape (n_samples, n_factors) or (n_factors,)
        actual_returns: Actual returns of shape (n_samples, n_factors) or (n_factors,)
        top_k: Number of top factors to consider (default 5)
        
    Returns:
        Dictionary with comprehensive metrics
    """
    # Convert to numpy arrays
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(actual_returns, torch.Tensor):
        actual_returns = actual_returns.detach().cpu().numpy()
    
    # Ensure 2D arrays
    if predictions.ndim == 1:
        predictions = predictions.reshape(1, -1)
    if actual_returns.ndim == 1:
        actual_returns = actual_returns.reshape(1, -1)
    
    n_samples, n_factors = predictions.shape
    
    # Initialize metrics containers
    top_k_returns = []
    hit_rates = []
    correlations_spearman = []
    correlations_pearson = []
    
    for i in range(n_samples):
        pred_sample = predictions[i]
        actual_sample = actual_returns[i]
        
        # Get top-k predicted and actual factor indices
        top_k_pred_idx = np.argsort(pred_sample)[-top_k:]
        top_k_actual_idx = np.argsort(actual_sample)[-top_k:]
        
        # Calculate average return of top-k predicted factors
        top_k_return = np.mean(actual_sample[top_k_pred_idx])
        top_k_returns.append(top_k_return)
        
        # Calculate hit rate (overlap between predicted and actual top-k)
        overlap = len(set(top_k_pred_idx).intersection(set(top_k_actual_idx)))
        hit_rate = overlap / top_k
        hit_rates.append(hit_rate)
        
        # Calculate correlations
        try:
            spearman_corr, _ = spearmanr(pred_sample, actual_sample)
            correlations_spearman.append(spearman_corr if not np.isnan(spearman_corr) else 0.0)
        except:
            correlations_spearman.append(0.0)
            
        try:
            pearson_corr, _ = pearsonr(pred_sample, actual_sample)
            correlations_pearson.append(pearson_corr if not np.isnan(pearson_corr) else 0.0)
        except:
            correlations_pearson.append(0.0)
    
    # Calculate averages
    avg_top_k_return = np.mean(top_k_returns)
    avg_hit_rate = np.mean(hit_rates)
    avg_spearman_corr = np.mean(correlations_spearman)
    avg_pearson_corr = np.mean(correlations_pearson)
    
    # Calculate Sharpe ratio (annualized, assuming monthly data)
    # Convert monthly returns to annualized Sharpe
    monthly_returns = np.array(top_k_returns)
    if len(monthly_returns) > 1:
        excess_returns = monthly_returns  # Assuming zero risk-free rate
        sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(12)
    else:
        sharpe_ratio = 0.0 if len(monthly_returns) == 0 else monthly_returns[0] * np.sqrt(12)
    
    # Information ratio (similar to Sharpe but against a benchmark)
    # Using zero return as benchmark
    if len(monthly_returns) > 1:
        information_ratio = np.mean(monthly_returns) / np.std(monthly_returns) * np.sqrt(12)
    else:
        information_ratio = sharpe_ratio
    
    # Calculate additional statistics
    return_volatility = np.std(top_k_returns) if len(top_k_returns) > 1 else 0.0
    hit_rate_std = np.std(hit_rates) if len(hit_rates) > 1 else 0.0
    
    # Downside deviation (for Sortino ratio)
    negative_returns = monthly_returns[monthly_returns < 0]
    downside_deviation = np.std(negative_returns) if len(negative_returns) > 1 else 0.0
    sortino_ratio = (np.mean(monthly_returns) / downside_deviation * np.sqrt(12)) if downside_deviation > 0 else float('inf')
    
    # Win rate (percentage of positive returns)
    win_rate = np.mean(monthly_returns > 0) if len(monthly_returns) > 0 else 0.0
    
    return {
        'top5_return': avg_top_k_return,
        'hit_rate': avg_hit_rate,
        'spearman_correlation': avg_spearman_corr,
        'pearson_correlation': avg_pearson_corr,
        'sharpe_ratio': sharpe_ratio,
        'information_ratio': information_ratio,
        'sortino_ratio': sortino_ratio,
        'return_volatility': return_volatility,
        'hit_rate_std': hit_rate_std,
        'win_rate': win_rate,
        'n_samples': n_samples,
        'top_k': top_k,
        'monthly_returns': monthly_returns.tolist(),
        'hit_rates_series': hit_rates
    }

def calculate_benchmark_metrics(actual_returns: Union[np.ndarray, torch.Tensor], 
                              top_k: int = 5) -> Dict[str, float]:
    """
    Calculate metrics for equal-weighted benchmark (random top-k selection).
    
    Args:
        actual_returns: Actual returns of shape (n_samples, n_factors) or (n_factors,)
        top_k: Number of top factors to consider
        
    Returns:
        Dictionary with benchmark metrics
    """
    # Convert to numpy arrays
    if isinstance(actual_returns, torch.Tensor):
        actual_returns = actual_returns.detach().cpu().numpy()
    
    # Ensure 2D arrays
    if actual_returns.ndim == 1:
        actual_returns = actual_returns.reshape(1, -1)
    
    n_samples, n_factors = actual_returns.shape
    
    # Equal-weighted benchmark: average return across all factors
    equal_weighted_returns = []
    random_top_k_returns = []
    
    np.random.seed(42)  # For reproducible benchmark
    
    for i in range(n_samples):
        actual_sample = actual_returns[i]
        
        # Equal-weighted return (all factors)
        equal_weighted_return = np.mean(actual_sample)
        equal_weighted_returns.append(equal_weighted_return)
        
        # Random top-k selection
        random_indices = np.random.choice(n_factors, size=top_k, replace=False)
        random_top_k_return = np.mean(actual_sample[random_indices])
        random_top_k_returns.append(random_top_k_return)
    
    # Calculate benchmark statistics
    ew_monthly_returns = np.array(equal_weighted_returns)
    random_monthly_returns = np.array(random_top_k_returns)
    
    # Equal-weighted benchmark metrics
    ew_sharpe = (np.mean(ew_monthly_returns) / np.std(ew_monthly_returns) * np.sqrt(12)) if np.std(ew_monthly_returns) > 0 else 0.0
    ew_win_rate = np.mean(ew_monthly_returns > 0)
    
    # Random top-k benchmark metrics
    random_sharpe = (np.mean(random_monthly_returns) / np.std(random_monthly_returns) * np.sqrt(12)) if np.std(random_monthly_returns) > 0 else 0.0
    random_win_rate = np.mean(random_monthly_returns > 0)
    
    return {
        'equal_weighted_return': np.mean(ew_monthly_returns),
        'equal_weighted_sharpe': ew_sharpe,
        'equal_weighted_volatility': np.std(ew_monthly_returns),
        'equal_weighted_win_rate': ew_win_rate,
        'random_top5_return': np.mean(random_monthly_returns),
        'random_top5_sharpe': random_sharpe,
        'random_top5_volatility': np.std(random_monthly_returns),
        'random_top5_win_rate': random_win_rate,
        'ew_monthly_returns': ew_monthly_returns.tolist(),
        'random_monthly_returns': random_monthly_returns.tolist()
    }

def compare_performance(model_metrics: Dict[str, float], 
                       benchmark_metrics: Dict[str, float]) -> Dict[str, float]:
    """
    Compare model performance against benchmarks.
    
    Args:
        model_metrics: Metrics from calculate_metrics()
        benchmark_metrics: Metrics from calculate_benchmark_metrics()
        
    Returns:
        Dictionary with performance comparison metrics
    """
    # Calculate excess returns and ratios
    excess_return_vs_ew = model_metrics['top5_return'] - benchmark_metrics['equal_weighted_return']
    excess_return_vs_random = model_metrics['top5_return'] - benchmark_metrics['random_top5_return']
    
    # Annualized excess returns
    annual_excess_vs_ew = excess_return_vs_ew * 12
    annual_excess_vs_random = excess_return_vs_random * 12
    
    # Return ratios
    return_ratio_vs_ew = model_metrics['top5_return'] / benchmark_metrics['equal_weighted_return'] if benchmark_metrics['equal_weighted_return'] != 0 else float('inf')
    return_ratio_vs_random = model_metrics['top5_return'] / benchmark_metrics['random_top5_return'] if benchmark_metrics['random_top5_return'] != 0 else float('inf')
    
    # Sharpe ratio comparisons
    sharpe_improvement_vs_ew = model_metrics['sharpe_ratio'] - benchmark_metrics['equal_weighted_sharpe']
    sharpe_improvement_vs_random = model_metrics['sharpe_ratio'] - benchmark_metrics['random_top5_sharpe']
    
    # Statistical significance (t-test)
    model_returns = np.array(model_metrics['monthly_returns'])
    ew_returns = np.array(benchmark_metrics['ew_monthly_returns'])
    random_returns = np.array(benchmark_metrics['random_monthly_returns'])
    
    # Ensure same length for comparison
    min_length = min(len(model_returns), len(ew_returns), len(random_returns))
    if min_length > 1:
        model_returns = model_returns[:min_length]
        ew_returns = ew_returns[:min_length]
        random_returns = random_returns[:min_length]
        
        try:
            t_stat_vs_ew, p_value_vs_ew = stats.ttest_rel(model_returns, ew_returns)
            t_stat_vs_random, p_value_vs_random = stats.ttest_rel(model_returns, random_returns)
        except:
            t_stat_vs_ew = p_value_vs_ew = 0.0
            t_stat_vs_random = p_value_vs_random = 0.0
    else:
        t_stat_vs_ew = p_value_vs_ew = 0.0
        t_stat_vs_random = p_value_vs_random = 0.0
    
    return {
        'excess_return_vs_equal_weighted': excess_return_vs_ew,
        'excess_return_vs_random_top5': excess_return_vs_random,
        'annual_excess_return_vs_ew': annual_excess_vs_ew,
        'annual_excess_return_vs_random': annual_excess_vs_random,
        'return_ratio_vs_equal_weighted': return_ratio_vs_ew,
        'return_ratio_vs_random_top5': return_ratio_vs_random,
        'sharpe_improvement_vs_ew': sharpe_improvement_vs_ew,
        'sharpe_improvement_vs_random': sharpe_improvement_vs_random,
        't_stat_vs_equal_weighted': t_stat_vs_ew,
        'p_value_vs_equal_weighted': p_value_vs_ew,
        't_stat_vs_random_top5': t_stat_vs_random,
        'p_value_vs_random_top5': p_value_vs_random,
        'is_significant_vs_ew': bool(p_value_vs_ew < 0.05),
        'is_significant_vs_random': bool(p_value_vs_random < 0.05)
    }

def calculate_rolling_metrics(monthly_returns: List[float], 
                            window_size: int = 12) -> Dict[str, List[float]]:
    """
    Calculate rolling performance metrics.
    
    Args:
        monthly_returns: List of monthly returns
        window_size: Rolling window size in months
        
    Returns:
        Dictionary with rolling metrics
    """
    if len(monthly_returns) < window_size:
        logger.warning(f"Not enough data for rolling metrics (need {window_size}, have {len(monthly_returns)})")
        return {
            'rolling_returns': [],
            'rolling_sharpe': [],
            'rolling_volatility': [],
            'rolling_win_rate': []
        }
    
    returns_array = np.array(monthly_returns)
    
    rolling_returns = []
    rolling_sharpe = []
    rolling_volatility = []
    rolling_win_rate = []
    
    for i in range(window_size, len(returns_array) + 1):
        window_returns = returns_array[i - window_size:i]
        
        # Rolling average return
        avg_return = np.mean(window_returns)
        rolling_returns.append(avg_return)
        
        # Rolling Sharpe ratio
        if np.std(window_returns) > 0:
            sharpe = avg_return / np.std(window_returns) * np.sqrt(12)
        else:
            sharpe = 0.0
        rolling_sharpe.append(sharpe)
        
        # Rolling volatility
        volatility = np.std(window_returns) * np.sqrt(12)
        rolling_volatility.append(volatility)
        
        # Rolling win rate
        win_rate = np.mean(window_returns > 0)
        rolling_win_rate.append(win_rate)
    
    return {
        'rolling_returns': rolling_returns,
        'rolling_sharpe': rolling_sharpe,
        'rolling_volatility': rolling_volatility,
        'rolling_win_rate': rolling_win_rate
    }

def create_performance_summary(model_metrics: Dict[str, float],
                             benchmark_metrics: Dict[str, float],
                             comparison_metrics: Dict[str, float]) -> str:
    """
    Create a formatted performance summary report.
    
    Args:
        model_metrics: Model performance metrics
        benchmark_metrics: Benchmark performance metrics
        comparison_metrics: Performance comparison metrics
        
    Returns:
        Formatted string report
    """
    summary = []
    summary.append("=" * 80)
    summary.append("FACTOR SELECTION MODEL PERFORMANCE SUMMARY")
    summary.append("=" * 80)
    summary.append("")
    
    # Model Performance
    summary.append("MODEL PERFORMANCE:")
    summary.append("-" * 40)
    summary.append(f"Average Top-5 Return:     {model_metrics['top5_return']:.4f} ({model_metrics['top5_return']*100:.2f}%)")
    summary.append(f"Hit Rate:                 {model_metrics['hit_rate']:.3f} ({model_metrics['hit_rate']*100:.1f}%)")
    summary.append(f"Sharpe Ratio:             {model_metrics['sharpe_ratio']:.3f}")
    summary.append(f"Information Ratio:        {model_metrics['information_ratio']:.3f}")
    summary.append(f"Sortino Ratio:            {model_metrics['sortino_ratio']:.3f}")
    summary.append(f"Win Rate:                 {model_metrics['win_rate']:.3f} ({model_metrics['win_rate']*100:.1f}%)")
    summary.append(f"Return Volatility:        {model_metrics['return_volatility']:.4f}")
    summary.append("")
    
    # Benchmark Performance
    summary.append("BENCHMARK PERFORMANCE:")
    summary.append("-" * 40)
    summary.append(f"Equal-Weighted Return:    {benchmark_metrics['equal_weighted_return']:.4f} ({benchmark_metrics['equal_weighted_return']*100:.2f}%)")
    summary.append(f"Equal-Weighted Sharpe:    {benchmark_metrics['equal_weighted_sharpe']:.3f}")
    summary.append(f"Random Top-5 Return:      {benchmark_metrics['random_top5_return']:.4f} ({benchmark_metrics['random_top5_return']*100:.2f}%)")
    summary.append(f"Random Top-5 Sharpe:      {benchmark_metrics['random_top5_sharpe']:.3f}")
    summary.append("")
    
    # Performance vs Benchmarks
    summary.append("PERFORMANCE vs BENCHMARKS:")
    summary.append("-" * 40)
    summary.append(f"Excess Return vs EW:      {comparison_metrics['excess_return_vs_equal_weighted']:.4f} ({comparison_metrics['annual_excess_return_vs_ew']:.2f}% annually)")
    summary.append(f"Return Ratio vs EW:       {comparison_metrics['return_ratio_vs_equal_weighted']:.2f}x")
    summary.append(f"Excess Return vs Random:  {comparison_metrics['excess_return_vs_random_top5']:.4f} ({comparison_metrics['annual_excess_return_vs_random']:.2f}% annually)")
    summary.append(f"Return Ratio vs Random:   {comparison_metrics['return_ratio_vs_random_top5']:.2f}x")
    summary.append("")
    
    # Statistical Significance
    summary.append("STATISTICAL SIGNIFICANCE:")
    summary.append("-" * 40)
    summary.append(f"vs Equal-Weighted:        {'Significant' if comparison_metrics['is_significant_vs_ew'] else 'Not Significant'} (p={comparison_metrics['p_value_vs_equal_weighted']:.4f})")
    summary.append(f"vs Random Top-5:          {'Significant' if comparison_metrics['is_significant_vs_random'] else 'Not Significant'} (p={comparison_metrics['p_value_vs_random_top5']:.4f})")
    summary.append("")
    
    # Success Criteria Check
    summary.append("SUCCESS CRITERIA CHECK:")
    summary.append("-" * 40)
    annual_excess_vs_ew = comparison_metrics['annual_excess_return_vs_ew']
    hit_rate = model_metrics['hit_rate']
    sharpe_ratio = model_metrics['sharpe_ratio']
    
    summary.append(f"✅ Beat EW by >10% annually: {'YES' if annual_excess_vs_ew > 0.10 else 'NO'} ({annual_excess_vs_ew:.1%})")
    summary.append(f"✅ Hit rate >30%:            {'YES' if hit_rate > 0.30 else 'NO'} ({hit_rate:.1%})")
    summary.append(f"✅ Sharpe ratio >1.0:        {'YES' if sharpe_ratio > 1.0 else 'NO'} ({sharpe_ratio:.2f})")
    summary.append("")
    
    summary.append(f"Sample Size: {model_metrics['n_samples']} months")
    summary.append("=" * 80)
    
    return "\n".join(summary)

def save_evaluation_results(model_metrics: Dict[str, float],
                          benchmark_metrics: Dict[str, float], 
                          comparison_metrics: Dict[str, float],
                          output_dir: str = OUTPUT_DIR) -> str:
    """
    Save comprehensive evaluation results to files.
    
    Args:
        model_metrics: Model performance metrics
        benchmark_metrics: Benchmark metrics
        comparison_metrics: Performance comparison metrics
        output_dir: Directory to save results
        
    Returns:
        Path to the saved summary report
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save detailed metrics to CSV
    all_metrics = {}
    all_metrics.update({f"model_{k}": v for k, v in model_metrics.items() if not isinstance(v, list)})
    all_metrics.update({f"benchmark_{k}": v for k, v in benchmark_metrics.items() if not isinstance(v, list)})
    all_metrics.update({f"comparison_{k}": v for k, v in comparison_metrics.items()})
    
    metrics_df = pd.DataFrame([all_metrics])
    metrics_path = os.path.join(output_dir, 'evaluation_metrics.csv')
    metrics_df.to_csv(metrics_path, index=False)
    
    # Save performance summary report
    summary_report = create_performance_summary(model_metrics, benchmark_metrics, comparison_metrics)
    summary_path = os.path.join(output_dir, 'performance_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(summary_report)
    
    logger.info(f"Evaluation results saved to {output_dir}")
    logger.info(f"Performance summary: {summary_path}")
    logger.info(f"Detailed metrics: {metrics_path}")
    
    return summary_path