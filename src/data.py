import pandas as pd
import numpy as np
import logging
import random
from typing import Tuple, List, Dict, Any
from .config import DATA_DIR, N_FACTORS

logger = logging.getLogger(__name__)

def load_data(t60_path: str = None, t2_path: str = None) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], pd.DatetimeIndex]:
    """
    Load forecast and actual return data from Excel files.
    
    Args:
        t60_path: Path to T60.xlsx (forecasted returns)
        t2_path: Path to T2_Optimizer.xlsx (actual returns)
    
    Returns:
        t60_df: DataFrame with forecasted returns
        t2_df: DataFrame with actual returns
        factor_names: List of factor names (columns)
        dates: DatetimeIndex of all dates
    """
    if t60_path is None:
        t60_path = f"{DATA_DIR}T60.xlsx"
    if t2_path is None:
        t2_path = f"{DATA_DIR}T2_Optimizer.xlsx"
    
    logger.info(f"Loading forecast data from {t60_path}")
    t60_df = pd.read_excel(t60_path, index_col=0, parse_dates=True)
    
    logger.info(f"Loading actual returns data from {t2_path}")
    t2_df = pd.read_excel(t2_path, index_col=0, parse_dates=True)
    
    # Validate data structure
    validate_data(t60_df, t2_df)
    
    factor_names = list(t60_df.columns)
    dates = t60_df.index
    
    logger.info(f"Loaded data: {len(dates)} months, {len(factor_names)} factors")
    logger.info(f"Date range: {dates[0]} to {dates[-1]}")
    
    return t60_df, t2_df, factor_names, dates

def validate_data(t60_df: pd.DataFrame, t2_df: pd.DataFrame) -> None:
    """
    Validate that both DataFrames meet requirements.
    
    Args:
        t60_df: Forecasted returns DataFrame
        t2_df: Actual returns DataFrame
    
    Raises:
        ValueError: If validation fails
    """
    logger.info("Validating data structure and alignment...")
    
    # Check shapes
    if t60_df.shape != t2_df.shape:
        raise ValueError(f"Shape mismatch: T60 {t60_df.shape} vs T2 {t2_df.shape}")
    
    # Check column count (updated to 83)
    if t60_df.shape[1] != N_FACTORS:
        logger.warning(f"Expected {N_FACTORS} factors, got {t60_df.shape[1]}")
    
    # Check column names match
    if not t60_df.columns.equals(t2_df.columns):
        raise ValueError("Column names don't match between files")
    
    # Check date alignment
    if not t60_df.index.equals(t2_df.index):
        raise ValueError("Date indices don't match between files")
    
    # Check for missing values
    if t60_df.isnull().any().any():
        raise ValueError("T60 contains missing values")
    if t2_df.isnull().any().any():
        raise ValueError("T2 contains missing values")
    
    # Check data types
    if not all(t60_df.dtypes == 'float64'):
        logger.warning("Not all T60 columns are float64")
    if not all(t2_df.dtypes == 'float64'):
        logger.warning("Not all T2 columns are float64")
    
    # Check for infinite values
    if np.isinf(t60_df.values).any():
        raise ValueError("T60 contains infinite values")
    if np.isinf(t2_df.values).any():
        raise ValueError("T2 contains infinite values")
    
    logger.info("✅ Data validation passed")

def create_rolling_windows(t60_df: pd.DataFrame, t2_df: pd.DataFrame, 
                          window_size: int = 60) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.Timestamp]]:
    """
    Create rolling windows for training and testing.
    
    Args:
        t60_df: Forecasted returns DataFrame
        t2_df: Actual returns DataFrame
        window_size: Number of months for training window
    
    Returns:
        List of tuples: [(train_X, train_y, test_X, test_y, test_date), ...]
        where train_X/test_X are forecast returns and train_y/test_y are actual returns
    """
    logger.info(f"Creating rolling windows with size {window_size}")
    
    windows = []
    n_periods = len(t60_df)
    
    # Start from window_size (need 60 months for first training)
    for i in range(window_size, n_periods):
        # Training data: previous window_size months
        train_start = i - window_size
        train_end = i
        
        train_X = t60_df.iloc[train_start:train_end].values  # Forecasted returns
        train_y = t2_df.iloc[train_start:train_end].values   # Actual returns
        
        # Test data: current month
        test_X = t60_df.iloc[i:i+1].values  # Single month forecast
        test_y = t2_df.iloc[i:i+1].values   # Single month actual
        test_date = t60_df.index[i]
        
        windows.append((train_X, train_y, test_X, test_y, test_date))
    
    logger.info(f"Created {len(windows)} rolling windows")
    logger.info(f"First test date: {windows[0][4]}")
    logger.info(f"Last test date: {windows[-1][4]}")
    
    return windows

def select_random_months(all_windows: List[Tuple], n_months: int = 30, seed: int = 42) -> List[Tuple]:
    """
    Select n_months random windows for hyperparameter tuning.
    
    Args:
        all_windows: List of all rolling windows
        n_months: Number of months to select
        seed: Random seed for reproducibility
    
    Returns:
        List of selected windows
    """
    logger.info(f"Selecting {n_months} random months for hyperparameter tuning (seed={seed})")
    
    random.seed(seed)
    np.random.seed(seed)
    
    if len(all_windows) < n_months:
        logger.warning(f"Only {len(all_windows)} windows available, using all")
        return all_windows
    
    selected_windows = random.sample(all_windows, n_months)
    
    # Log selected dates for transparency
    selected_dates = [window[4] for window in selected_windows]
    logger.info(f"Selected dates: {min(selected_dates)} to {max(selected_dates)}")
    
    return selected_windows

def get_data_statistics(t60_df: pd.DataFrame, t2_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate data statistics for analysis.
    
    Returns:
        Dictionary with statistics
    """
    stats = {
        'n_months': len(t60_df),
        'n_factors': len(t60_df.columns),
        'date_range': (t60_df.index[0], t60_df.index[-1]),
        't60_stats': {
            'mean': t60_df.mean().mean(),
            'std': t60_df.std().mean(),
            'min': t60_df.min().min(),
            'max': t60_df.max().max()
        },
        't2_stats': {
            'mean': t2_df.mean().mean(),
            'std': t2_df.std().mean(),
            'min': t2_df.min().min(),
            'max': t2_df.max().max()
        }
    }
    return stats

def prepare_data_for_training(train_X: np.ndarray, train_y: np.ndarray, 
                             val_split: float = 0.2, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split training data into train/validation sets.
    
    Args:
        train_X: Training features (forecasted returns)
        train_y: Training targets (actual returns)
        val_split: Fraction for validation
        seed: Random seed
    
    Returns:
        train_X_split, train_y_split, val_X, val_y
    """
    np.random.seed(seed)
    
    n_samples = len(train_X)
    n_val = int(n_samples * val_split)
    
    # Random indices for validation
    val_indices = np.random.choice(n_samples, n_val, replace=False)
    train_indices = np.setdiff1d(np.arange(n_samples), val_indices)
    
    train_X_split = train_X[train_indices]
    train_y_split = train_y[train_indices]
    val_X = train_X[val_indices]
    val_y = train_y[val_indices]
    
    return train_X_split, train_y_split, val_X, val_y