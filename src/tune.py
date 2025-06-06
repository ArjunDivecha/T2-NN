import os
import time
import logging
import itertools
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import torch

from .config import HYPERPARAM_GRID, OUTPUT_DIR, get_device, setup_logging
from .data import load_data, create_rolling_windows, select_random_months
from .train import train_model
from .model import calculate_top5_metrics

logger = logging.getLogger(__name__)

def generate_hyperparameter_combinations() -> List[Dict[str, Any]]:
    """
    Generate all combinations of hyperparameters from the grid.
    
    Returns:
        List of hyperparameter dictionaries
    """
    # Get all parameter names and their possible values
    param_names = list(HYPERPARAM_GRID.keys())
    param_values = list(HYPERPARAM_GRID.values())
    
    # Generate all combinations
    combinations = []
    for combination in itertools.product(*param_values):
        param_dict = dict(zip(param_names, combination))
        combinations.append(param_dict)
    
    logger.info(f"Generated {len(combinations)} hyperparameter combinations")
    return combinations

def evaluate_hyperparameters(params: Dict[str, Any], train_data: List[Tuple], 
                           device: torch.device) -> Dict[str, float]:
    """
    Evaluate a single hyperparameter configuration on training data.
    
    Args:
        params: Hyperparameter dictionary
        train_data: List of (train_X, train_y, val_X, val_y, date) tuples
        device: Device to train on
        
    Returns:
        Dictionary with evaluation metrics
    """
    logger.info(f"Evaluating hyperparameters: {params}")
    
    # Track metrics across all training windows
    all_returns = []
    all_hit_rates = []
    all_train_times = []
    all_epochs = []
    
    for i, (train_X, train_y, val_X, val_y, date) in enumerate(train_data):
        try:
            # Create training configuration
            config = params.copy()
            config.update({
                'n_epochs': 100,
                'early_stopping_patience': 10,
                'val_split': 0.0,  # Use provided validation data
                'random_seed': 42
            })
            
            start_time = time.time()
            
            # Train model
            model, history = train_model(train_X, train_y, val_X, val_y, config)
            
            train_time = time.time() - start_time
            
            # Evaluate on validation set
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
    
    # Calculate average metrics
    avg_return = np.mean(all_returns)
    avg_hit_rate = np.mean(all_hit_rates)
    avg_train_time = np.mean(all_train_times)
    avg_epochs = np.mean(all_epochs)
    
    # Calculate stability metrics
    return_std = np.std(all_returns)
    hit_rate_std = np.std(all_hit_rates)
    
    results = {
        'avg_top5_return': avg_return,
        'avg_hit_rate': avg_hit_rate,
        'avg_train_time': avg_train_time,
        'avg_epochs': avg_epochs,
        'return_std': return_std,
        'hit_rate_std': hit_rate_std,
        'success_rate': sum(1 for r in all_returns if r > -0.05) / len(all_returns)  # Fraction of successful trainings
    }
    
    logger.info(f"Results - Return: {avg_return:.4f}±{return_std:.4f}, "
                f"Hit Rate: {avg_hit_rate:.3f}±{hit_rate_std:.3f}, "
                f"Time: {avg_train_time:.2f}s")
    
    return results

def run_hyperparameter_search(data_dir: str = 'data/', n_months: int = 30, 
                             save_results: bool = True) -> pd.DataFrame:
    """
    Run comprehensive hyperparameter search.
    
    Args:
        data_dir: Directory containing data files
        n_months: Number of random months to use for tuning
        save_results: Whether to save results to CSV
        
    Returns:
        DataFrame with results for all hyperparameter combinations
    """
    logger.info("Starting hyperparameter search")
    start_time = time.time()
    
    # Load data
    logger.info("Loading data...")
    t60_path = os.path.join(data_dir, 'T60.xlsx')
    t2_path = os.path.join(data_dir, 'T2_Optimizer.xlsx')
    
    t60_df, t2_df, factor_names, dates = load_data(t60_path, t2_path)
    logger.info(f"Loaded data: {len(dates)} months, {len(factor_names)} factors")
    
    # Create rolling windows
    logger.info("Creating rolling windows...")
    all_windows = create_rolling_windows(t60_df, t2_df, window_size=60)
    logger.info(f"Created {len(all_windows)} rolling windows")
    
    # Select random months for tuning
    logger.info(f"Selecting {n_months} random months for hyperparameter tuning...")
    selected_windows = select_random_months(all_windows, n_months=n_months, seed=42)
    logger.info(f"Selected {len(selected_windows)} months for tuning")
    
    # Get device
    device = get_device()
    logger.info(f"Using device: {device}")
    
    # Generate hyperparameter combinations
    param_combinations = generate_hyperparameter_combinations()
    logger.info(f"Testing {len(param_combinations)} hyperparameter combinations")
    
    # Track results
    results = []
    
    for i, params in enumerate(param_combinations):
        logger.info(f"Progress: {i+1}/{len(param_combinations)} - Testing: {params}")
        
        try:
            # Evaluate this parameter combination
            metrics = evaluate_hyperparameters(params, selected_windows, device)
            
            # Combine parameters and results
            result_row = params.copy()
            result_row.update(metrics)
            results.append(result_row)
            
            logger.info(f"Completed {i+1}/{len(param_combinations)} - "
                       f"Return: {metrics['avg_top5_return']:.4f}")
            
        except Exception as e:
            logger.error(f"Failed to evaluate parameters {params}: {e}")
            # Record failed attempt
            result_row = params.copy()
            result_row.update({
                'avg_top5_return': -1.0,
                'avg_hit_rate': 0.0,
                'avg_train_time': 0.0,
                'avg_epochs': 0,
                'return_std': 0.0,
                'hit_rate_std': 0.0,
                'success_rate': 0.0
            })
            results.append(result_row)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Sort by primary metric (average top-5 return)
    results_df = results_df.sort_values('avg_top5_return', ascending=False)
    
    total_time = time.time() - start_time
    logger.info(f"Hyperparameter search completed in {total_time:.2f}s")
    
    # Log top results
    logger.info("Top 5 hyperparameter combinations:")
    for i, (_, row) in enumerate(results_df.head().iterrows()):
        logger.info(f"  {i+1}. Return: {row['avg_top5_return']:.4f}, "
                   f"Hit Rate: {row['avg_hit_rate']:.3f}, "
                   f"Hidden: {row['hidden_sizes']}, "
                   f"LR: {row['learning_rate']:.0e}, "
                   f"Dropout: {row['dropout_rate']:.1f}")
    
    # Save results
    if save_results:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        results_path = os.path.join(OUTPUT_DIR, 'hyperparam_results.csv')
        results_df.to_csv(results_path, index=False)
        logger.info(f"Results saved to {results_path}")
    
    return results_df

def get_best_hyperparameters(results_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Extract the best hyperparameter configuration from results.
    
    Args:
        results_df: DataFrame with hyperparameter search results
        
    Returns:
        Dictionary with best hyperparameters
    """
    best_row = results_df.iloc[0]  # Already sorted by performance
    
    best_params = {
        'hidden_sizes': best_row['hidden_sizes'],
        'learning_rate': best_row['learning_rate'],
        'dropout_rate': best_row['dropout_rate'],
        'batch_size': best_row['batch_size'],
        'weight_decay': best_row['weight_decay']
    }
    
    logger.info("Best hyperparameters found:")
    for key, value in best_params.items():
        logger.info(f"  {key}: {value}")
    logger.info(f"  Performance: {best_row['avg_top5_return']:.4f} return, "
                f"{best_row['avg_hit_rate']:.3f} hit rate")
    
    return best_params

def update_default_config(best_params: Dict[str, Any], config_file: str = 'src/config.py'):
    """
    Update the default configuration with the best hyperparameters.
    
    Args:
        best_params: Best hyperparameter configuration
        config_file: Path to config file to update
    """
    logger.info("Updating default configuration with best hyperparameters")
    
    # Read current config file
    with open(config_file, 'r') as f:
        content = f.read()
    
    # Create new DEFAULT_CONFIG section
    new_config_section = f"""# Model defaults (updated from hyperparameter tuning)
DEFAULT_CONFIG = {{
    'hidden_sizes': {best_params['hidden_sizes']},
    'learning_rate': {best_params['learning_rate']},
    'dropout_rate': {best_params['dropout_rate']},
    'batch_size': {best_params['batch_size']},
    'weight_decay': {best_params['weight_decay']}
}}"""
    
    # Replace the DEFAULT_CONFIG section
    import re
    pattern = r'# Model defaults.*?DEFAULT_CONFIG = \{[^}]*\}'
    if re.search(pattern, content, re.DOTALL):
        content = re.sub(pattern, new_config_section, content, flags=re.DOTALL)
    else:
        # If pattern not found, append at the end
        content += '\n\n' + new_config_section
    
    # Write back to file
    with open(config_file, 'w') as f:
        f.write(content)
    
    logger.info(f"Updated {config_file} with best hyperparameters")

def main():
    """Main function for hyperparameter tuning."""
    # Setup logging
    setup_logging()
    logger.info("Starting hyperparameter tuning pipeline")
    
    try:
        # Run hyperparameter search
        results_df = run_hyperparameter_search(
            data_dir='data/',
            n_months=30,
            save_results=True
        )
        
        # Get best parameters
        best_params = get_best_hyperparameters(results_df)
        
        # Update default configuration
        update_default_config(best_params)
        
        logger.info("Hyperparameter tuning completed successfully!")
        
        return results_df, best_params
        
    except Exception as e:
        logger.error(f"Hyperparameter tuning failed: {e}")
        raise

if __name__ == "__main__":
    main()