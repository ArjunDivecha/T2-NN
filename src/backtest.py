import os
import time
import logging
import pandas as pd
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from functools import partial

from .config import get_device, DEFAULT_CONFIG, TRAINING_CONFIG, BACKTEST_CONFIG, OUTPUT_DIR
from .data import load_data, create_rolling_windows
from .train import train_model, save_monthly_forecasts, save_rolling_forecasts
from .evaluate import calculate_metrics, calculate_benchmark_metrics, compare_performance, save_evaluation_results
from .model import calculate_top5_metrics

logger = logging.getLogger(__name__)

def train_and_predict_month(window_data: Tuple, config: Dict, device_str: str = 'mps') -> Dict:
    """
    Train model and make prediction for a single month.
    This function is designed to be used with multiprocessing.
    
    Args:
        window_data: Tuple of (train_X, train_y, test_X, test_y, test_date)
        config: Training configuration
        device_str: Device string ('mps', 'cpu')
        
    Returns:
        Dictionary with prediction results
    """
    try:
        train_X, train_y, test_X, test_y, test_date = window_data
        
        # Set device (need to do this in each process)
        device = torch.device(device_str if device_str == 'mps' and torch.backends.mps.is_available() else 'cpu')
        
        start_time = time.time()
        
        # Train model
        model, history = train_model(train_X, train_y, config=config)
        
        train_time = time.time() - start_time
        
        # Make prediction
        model.eval()
        with torch.no_grad():
            test_X_tensor = torch.FloatTensor(test_X).to(device)
            predictions = model(test_X_tensor).cpu().numpy()
            
            # For single sample prediction
            if predictions.ndim == 2 and predictions.shape[0] == 1:
                predictions = predictions.flatten()
            
            test_y_flat = test_y.flatten() if test_y.ndim > 1 else test_y
        
        # Calculate metrics
        metrics = calculate_top5_metrics(
            torch.FloatTensor(predictions.reshape(1, -1)), 
            torch.FloatTensor(test_y_flat.reshape(1, -1))
        )
        
        # Get top 5 factor indices
        top5_indices = np.argsort(predictions)[-5:]
        top5_returns = test_y_flat[top5_indices]
        portfolio_return = np.mean(top5_returns)
        
        return {
            'date': test_date,
            'predictions': predictions.tolist(),
            'actual_returns': test_y_flat.tolist(),
            'top5_indices': top5_indices.tolist(),
            'top5_returns': top5_returns.tolist(),
            'portfolio_return': portfolio_return,
            'hit_rate': metrics['hit_rate'],
            'avg_top5_return': metrics['avg_top5_return'],
            'train_time': train_time,
            'final_epoch': history['final_epoch'],
            'best_val_return': history['best_val_return'],
            'success': True
        }
        
    except Exception as e:
        logger.error(f"Failed to process month {test_date}: {e}")
        return {
            'date': test_date,
            'predictions': None,
            'actual_returns': None,
            'top5_indices': None,
            'top5_returns': None,
            'portfolio_return': None,
            'hit_rate': 0.0,
            'avg_top5_return': 0.0,
            'train_time': 0.0,
            'final_epoch': 0,
            'best_val_return': 0.0,
            'success': False,
            'error': str(e)
        }

def run_backtest_sequential(data_dir: str = 'data/', 
                           config: Optional[Dict] = None,
                           start_date: Optional[str] = None,
                           end_date: Optional[str] = None,
                           save_forecasts: bool = True) -> pd.DataFrame:
    """
    Run sequential backtest (single-threaded).
    
    Args:
        data_dir: Directory containing data files
        config: Training configuration
        start_date: Start date for backtest (YYYY-MM-DD format)
        end_date: End date for backtest (YYYY-MM-DD format)
        save_forecasts: Whether to save individual monthly forecasts
        
    Returns:
        DataFrame with backtest results
    """
    logger.info("Starting sequential backtest")
    
    # Load data
    t60_path = os.path.join(data_dir, 'T60.xlsx')
    t2_path = os.path.join(data_dir, 'T2_Optimizer.xlsx')
    
    t60_df, t2_df, factor_names, dates = load_data(t60_path, t2_path)
    
    # Create rolling windows
    all_windows = create_rolling_windows(t60_df, t2_df, window_size=BACKTEST_CONFIG['window_size'])
    
    # Filter by date range if specified
    if start_date or end_date:
        filtered_windows = []
        for window in all_windows:
            test_date = window[4]
            if start_date and test_date < pd.Timestamp(start_date):
                continue
            if end_date and test_date > pd.Timestamp(end_date):
                continue
            filtered_windows.append(window)
        all_windows = filtered_windows
    
    logger.info(f"Running backtest on {len(all_windows)} months")
    
    # Use default config if none provided
    if config is None:
        config = DEFAULT_CONFIG.copy()
        config.update(TRAINING_CONFIG)
    
    # Get device
    device = get_device()
    device_str = str(device)
    
    results = []
    monthly_forecasts = []
    
    for i, window in enumerate(all_windows):
        test_date = window[4]
        logger.info(f"Processing month {i+1}/{len(all_windows)}: {test_date}")
        
        # Train and predict
        result = train_and_predict_month(window, config, device_str)
        results.append(result)
        
        # Save monthly forecast if requested
        if save_forecasts and result['success'] and result['predictions']:
            try:
                # Create model for saving forecasts (simplified)
                predictions = np.array(result['predictions'])
                forecast_df = pd.DataFrame({
                    'Date': [test_date],
                    **{factor_names[j]: [predictions[j]] for j in range(len(factor_names))}
                })
                monthly_forecasts.append(forecast_df)
            except Exception as e:
                logger.warning(f"Failed to save forecast for {test_date}: {e}")
        
        # Log progress
        if (i + 1) % 10 == 0:
            success_rate = sum(1 for r in results if r['success']) / len(results)
            avg_return = np.mean([r['portfolio_return'] for r in results if r['success']])
            logger.info(f"Progress: {i+1}/{len(all_windows)}, Success rate: {success_rate:.1%}, Avg return: {avg_return:.4f}")
    
    # Save all forecasts
    if save_forecasts and monthly_forecasts:
        try:
            output_file = os.path.join(OUTPUT_DIR, 'T60_Enhanced_NN.xlsx')
            save_rolling_forecasts(monthly_forecasts, output_file)
        except Exception as e:
            logger.error(f"Failed to save rolling forecasts: {e}")
    
    # Convert results to DataFrame
    results_df = create_backtest_dataframe(results, factor_names)
    
    logger.info(f"Sequential backtest completed: {len(results_df)} months processed")
    
    return results_df

def run_backtest_parallel(data_dir: str = 'data/', 
                         config: Optional[Dict] = None,
                         start_date: Optional[str] = None,
                         end_date: Optional[str] = None,
                         n_jobs: Optional[int] = None,
                         save_forecasts: bool = True) -> pd.DataFrame:
    """
    Run parallel backtest using multiprocessing.
    
    Args:
        data_dir: Directory containing data files
        config: Training configuration
        start_date: Start date for backtest
        end_date: End date for backtest
        n_jobs: Number of parallel jobs (default: min(8, cpu_count()))
        save_forecasts: Whether to save individual monthly forecasts
        
    Returns:
        DataFrame with backtest results
    """
    logger.info("Starting parallel backtest")
    
    # Load data
    t60_path = os.path.join(data_dir, 'T60.xlsx')
    t2_path = os.path.join(data_dir, 'T2_Optimizer.xlsx')
    
    t60_df, t2_df, factor_names, dates = load_data(t60_path, t2_path)
    
    # Create rolling windows
    all_windows = create_rolling_windows(t60_df, t2_df, window_size=BACKTEST_CONFIG['window_size'])
    
    # Filter by date range if specified
    if start_date or end_date:
        filtered_windows = []
        for window in all_windows:
            test_date = window[4]
            if start_date and test_date < pd.Timestamp(start_date):
                continue
            if end_date and test_date > pd.Timestamp(end_date):
                continue
            filtered_windows.append(window)
        all_windows = filtered_windows
    
    logger.info(f"Running parallel backtest on {len(all_windows)} months")
    
    # Use default config if none provided
    if config is None:
        config = DEFAULT_CONFIG.copy()
        config.update(TRAINING_CONFIG)
    
    # Determine number of jobs
    if n_jobs is None:
        n_jobs = min(BACKTEST_CONFIG['n_jobs'], len(all_windows), multiprocessing.cpu_count())
    
    logger.info(f"Using {n_jobs} parallel processes")
    
    # Create partial function with fixed config
    # Note: Use 'cpu' for parallel processing to avoid MPS issues
    train_predict_func = partial(train_and_predict_month, config=config, device_str='cpu')
    
    results = []
    completed = 0
    
    # Run parallel processing
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        # Submit all jobs
        future_to_window = {executor.submit(train_predict_func, window): window for window in all_windows}
        
        # Collect results as they complete
        for future in as_completed(future_to_window):
            try:
                result = future.result()
                results.append(result)
                completed += 1
                
                # Log progress
                if completed % 10 == 0 or completed == len(all_windows):
                    success_rate = sum(1 for r in results if r['success']) / len(results)
                    avg_return = np.mean([r['portfolio_return'] for r in results if r['success'] and r['portfolio_return'] is not None])
                    logger.info(f"Progress: {completed}/{len(all_windows)}, Success rate: {success_rate:.1%}, Avg return: {avg_return:.4f}")
                    
            except Exception as e:
                logger.error(f"Parallel processing error: {e}")
                # Add a failed result
                window = future_to_window[future]
                results.append({
                    'date': window[4],
                    'success': False,
                    'error': str(e),
                    'predictions': None,
                    'actual_returns': None,
                    'top5_indices': None,
                    'top5_returns': None,
                    'portfolio_return': None,
                    'hit_rate': 0.0,
                    'avg_top5_return': 0.0,
                    'train_time': 0.0,
                    'final_epoch': 0,
                    'best_val_return': 0.0
                })
                completed += 1
    
    # Sort results by date
    results.sort(key=lambda x: x['date'])
    
    # Save forecasts if requested
    if save_forecasts:
        monthly_forecasts = []
        for result in results:
            if result['success'] and result['predictions']:
                try:
                    predictions = np.array(result['predictions'])
                    forecast_df = pd.DataFrame({
                        'Date': [result['date']],
                        **{factor_names[j]: [predictions[j]] for j in range(len(factor_names))}
                    })
                    monthly_forecasts.append(forecast_df)
                except Exception as e:
                    logger.warning(f"Failed to create forecast for {result['date']}: {e}")
        
        if monthly_forecasts:
            try:
                output_file = os.path.join(OUTPUT_DIR, 'T60_Enhanced_NN.xlsx')
                save_rolling_forecasts(monthly_forecasts, output_file)
            except Exception as e:
                logger.error(f"Failed to save rolling forecasts: {e}")
    
    # Convert results to DataFrame
    results_df = create_backtest_dataframe(results, factor_names)
    
    logger.info(f"Parallel backtest completed: {len(results_df)} months processed")
    
    return results_df

def create_backtest_dataframe(results: List[Dict], factor_names: List[str]) -> pd.DataFrame:
    """
    Convert backtest results to a structured DataFrame.
    
    Args:
        results: List of result dictionaries
        factor_names: List of factor names
        
    Returns:
        DataFrame with backtest results
    """
    # Filter successful results
    successful_results = [r for r in results if r['success']]
    
    if not successful_results:
        logger.warning("No successful results to convert to DataFrame")
        return pd.DataFrame()
    
    # Create basic columns
    data = {
        'date': [r['date'] for r in successful_results],
        'portfolio_return': [r['portfolio_return'] for r in successful_results],
        'hit_rate': [r['hit_rate'] for r in successful_results],
        'avg_top5_return': [r['avg_top5_return'] for r in successful_results],
        'train_time': [r['train_time'] for r in successful_results],
        'final_epoch': [r['final_epoch'] for r in successful_results],
        'best_val_return': [r['best_val_return'] for r in successful_results]
    }
    
    # Add top 5 factor names
    for i in range(5):
        factor_col = f'top5_factor_{i+1}'
        data[factor_col] = []
        
        for r in successful_results:
            if r['top5_indices'] and len(r['top5_indices']) > i:
                factor_idx = r['top5_indices'][i]
                if factor_idx < len(factor_names):
                    data[factor_col].append(factor_names[factor_idx])
                else:
                    data[factor_col].append(f'Factor_{factor_idx}')
            else:
                data[factor_col].append(None)
    
    # Add top 5 returns
    for i in range(5):
        return_col = f'top5_return_{i+1}'
        data[return_col] = []
        
        for r in successful_results:
            if r['top5_returns'] and len(r['top5_returns']) > i:
                data[return_col].append(r['top5_returns'][i])
            else:
                data[return_col].append(None)
    
    df = pd.DataFrame(data)
    df = df.sort_values('date').reset_index(drop=True)
    
    return df

def analyze_backtest_results(results_df: pd.DataFrame) -> Dict:
    """
    Analyze backtest results and calculate comprehensive performance metrics.
    
    Args:
        results_df: DataFrame with backtest results
        
    Returns:
        Dictionary with performance analysis
    """
    if results_df.empty:
        logger.warning("Empty results DataFrame for analysis")
        return {}
    
    # Basic statistics
    n_months = len(results_df)
    portfolio_returns = results_df['portfolio_return'].values
    hit_rates = results_df['hit_rate'].values
    train_times = results_df['train_time'].values
    
    # Calculate performance metrics
    avg_return = np.mean(portfolio_returns)
    return_std = np.std(portfolio_returns)
    avg_hit_rate = np.mean(hit_rates)
    hit_rate_std = np.std(hit_rates)
    
    # Annualized metrics
    annual_return = avg_return * 12
    annual_volatility = return_std * np.sqrt(12)
    sharpe_ratio = avg_return / return_std * np.sqrt(12) if return_std > 0 else 0.0
    
    # Additional metrics
    win_rate = np.mean(portfolio_returns > 0)
    max_return = np.max(portfolio_returns)
    min_return = np.min(portfolio_returns)
    
    # Drawdown analysis
    cumulative_returns = np.cumprod(1 + portfolio_returns)
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdowns = (cumulative_returns - running_max) / running_max
    max_drawdown = np.min(drawdowns)
    
    # Training efficiency
    avg_train_time = np.mean(train_times)
    total_train_time = np.sum(train_times)
    avg_epochs = np.mean(results_df['final_epoch'].values)
    
    # Rolling performance (if enough data)
    rolling_12m_returns = []
    rolling_12m_sharpe = []
    rolling_12m_hit_rates = []
    
    if n_months >= 12:
        for i in range(12, n_months + 1):
            window_returns = portfolio_returns[i-12:i]
            window_hit_rates = hit_rates[i-12:i]
            
            rolling_return = np.mean(window_returns)
            rolling_vol = np.std(window_returns)
            rolling_sharpe = rolling_return / rolling_vol * np.sqrt(12) if rolling_vol > 0 else 0.0
            rolling_hit_rate = np.mean(window_hit_rates)
            
            rolling_12m_returns.append(rolling_return)
            rolling_12m_sharpe.append(rolling_sharpe)
            rolling_12m_hit_rates.append(rolling_hit_rate)
    
    return {
        # Basic metrics
        'n_months': n_months,
        'avg_monthly_return': avg_return,
        'monthly_return_std': return_std,
        'avg_hit_rate': avg_hit_rate,
        'hit_rate_std': hit_rate_std,
        
        # Annualized metrics
        'annual_return': annual_return,
        'annual_volatility': annual_volatility,
        'sharpe_ratio': sharpe_ratio,
        
        # Risk metrics
        'win_rate': win_rate,
        'max_return': max_return,
        'min_return': min_return,
        'max_drawdown': max_drawdown,
        
        # Training metrics
        'avg_train_time': avg_train_time,
        'total_train_time': total_train_time,
        'avg_epochs': avg_epochs,
        
        # Rolling metrics
        'rolling_12m_returns': rolling_12m_returns,
        'rolling_12m_sharpe': rolling_12m_sharpe,
        'rolling_12m_hit_rates': rolling_12m_hit_rates,
        
        # Time series
        'dates': results_df['date'].tolist(),
        'monthly_returns': portfolio_returns.tolist(),
        'monthly_hit_rates': hit_rates.tolist(),
        'cumulative_returns': cumulative_returns.tolist(),
        'drawdowns': drawdowns.tolist()
    }

def save_backtest_results(results_df: pd.DataFrame, analysis: Dict, 
                         output_dir: str = OUTPUT_DIR) -> Tuple[str, str]:
    """
    Save backtest results and analysis to files.
    
    Args:
        results_df: DataFrame with backtest results
        analysis: Performance analysis dictionary
        output_dir: Directory to save results
        
    Returns:
        Tuple of (results_file_path, analysis_file_path)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save detailed results
    results_path = os.path.join(output_dir, 'backtest_results.csv')
    results_df.to_csv(results_path, index=False)
    
    # Save analysis summary
    analysis_path = os.path.join(output_dir, 'backtest_analysis.csv')
    
    # Convert analysis to DataFrame (excluding lists)
    analysis_simple = {k: v for k, v in analysis.items() 
                      if not isinstance(v, list) and not isinstance(v, np.ndarray)}
    analysis_df = pd.DataFrame([analysis_simple])
    analysis_df.to_csv(analysis_path, index=False)
    
    logger.info(f"Backtest results saved to {results_path}")
    logger.info(f"Backtest analysis saved to {analysis_path}")
    
    return results_path, analysis_path

def run_backtest(data_dir: str = 'data/', 
                parallel: bool = True,
                start_date: Optional[str] = None,
                end_date: Optional[str] = None,
                config: Optional[Dict] = None,
                save_results: bool = True,
                save_forecasts: bool = True) -> Tuple[pd.DataFrame, Dict]:
    """
    Main backtest function that runs the complete rolling window backtest.
    
    Args:
        data_dir: Directory containing data files
        parallel: Whether to use parallel processing
        start_date: Start date for backtest (YYYY-MM-DD)
        end_date: End date for backtest (YYYY-MM-DD)
        config: Training configuration (uses default if None)
        save_results: Whether to save results to files
        save_forecasts: Whether to save monthly forecasts
        
    Returns:
        Tuple of (results_DataFrame, analysis_dict)
    """
    logger.info(f"Starting {'parallel' if parallel else 'sequential'} backtest")
    start_time = time.time()
    
    # Run backtest
    if parallel:
        results_df = run_backtest_parallel(
            data_dir=data_dir,
            config=config,
            start_date=start_date,
            end_date=end_date,
            save_forecasts=save_forecasts
        )
    else:
        results_df = run_backtest_sequential(
            data_dir=data_dir,
            config=config,
            start_date=start_date,
            end_date=end_date,
            save_forecasts=save_forecasts
        )
    
    # Analyze results
    analysis = analyze_backtest_results(results_df)
    
    total_time = time.time() - start_time
    analysis['total_backtest_time'] = total_time
    
    # Save results
    if save_results:
        save_backtest_results(results_df, analysis)
    
    # Log summary
    logger.info(f"Backtest completed in {total_time:.2f}s")
    if analysis and 'n_months' in analysis:
        logger.info(f"Processed {analysis['n_months']} months")
        logger.info(f"Average monthly return: {analysis['avg_monthly_return']:.4f}")
        logger.info(f"Annual return: {analysis['annual_return']:.2%}")
        logger.info(f"Sharpe ratio: {analysis['sharpe_ratio']:.3f}")
        logger.info(f"Average hit rate: {analysis['avg_hit_rate']:.3f}")
    else:
        logger.warning("No successful backtest results to analyze")
    
    return results_df, analysis