#!/usr/bin/env python3
"""
Run a single window properly and show the results.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import torch
import logging
from src.data import load_data, create_rolling_windows
from src.model import SimpleNN
from src.train import calculate_top5_metrics
from src.config import get_device

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_single_window(window_index=119):  # 2015-01-01 window
    """
    Run a single window with the CORRECT methodology:
    1. Train on ALL 60 months (no validation split)
    2. Predict the next month
    3. Show results
    """
    logger.info(f"Running single window {window_index} with CORRECT methodology")
    
    # Load data
    forecast_df, actual_df, factor_names, dates = load_data()
    rolling_windows = create_rolling_windows(forecast_df, actual_df, window_size=60)
    
    if window_index >= len(rolling_windows):
        logger.error(f"Window index {window_index} out of range. Max: {len(rolling_windows)-1}")
        return
    
    # Get the window data
    train_X, train_y, target_X, target_y, target_date = rolling_windows[window_index]
    
    logger.info("="*80)
    logger.info(f"WINDOW {window_index} - CORRECT IMPLEMENTATION")
    logger.info("="*80)
    
    # Show window details
    target_date_idx = dates.get_loc(target_date)
    train_start_idx = target_date_idx - 60
    train_end_idx = target_date_idx
    train_dates = dates[train_start_idx:train_end_idx]
    
    logger.info(f"Target month (predicting): {target_date}")
    logger.info(f"Training period: {train_dates[0]} to {train_dates[-1]} ({len(train_dates)} months)")
    logger.info(f"Training data shape: X={train_X.shape}, y={train_y.shape}")
    logger.info(f"Target data shape: X={target_X.shape}, y={target_y.shape}")
    
    # Configuration (using the "best" hyperparameters from original tuning)
    config = {
        'hidden_sizes': [512, 256],
        'learning_rate': 0.001,
        'dropout_rate': 0.4,
        'batch_size': 32,
        'weight_decay': 1e-5,
        'n_epochs': 100,
        'early_stopping_patience': 10,
        'random_seed': 42
    }
    
    logger.info(f"Using configuration: {config}")
    
    # Setup device
    device = get_device()
    logger.info(f"Using device: {device}")
    
    # Create model
    n_factors = train_X.shape[1]
    model = SimpleNN(
        input_size=n_factors,
        hidden_sizes=config['hidden_sizes'],
        dropout_rate=config['dropout_rate']
    ).to(device)
    
    logger.info(f"Created model: {n_factors} → {' → '.join(map(str, config['hidden_sizes']))} → {n_factors}")
    logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Setup training
    optimizer = torch.optim.Adam(model.parameters(), 
                                lr=config['learning_rate'], 
                                weight_decay=config['weight_decay'])
    
    # Convert to tensors
    train_X_tensor = torch.FloatTensor(train_X).to(device)
    train_y_tensor = torch.FloatTensor(train_y).to(device)
    target_X_tensor = torch.FloatTensor(target_X).to(device)
    target_y_tensor = torch.FloatTensor(target_y).to(device)
    
    # Custom top-5 loss function
    def top5_return_loss(predictions, actual_returns):
        batch_size = predictions.shape[0]
        total_loss = 0.0
        
        for i in range(batch_size):
            pred_i = predictions[i]
            actual_i = actual_returns[i]
            
            # Get top 5 predictions using temperature-scaled softmax
            temperature = 0.1
            top5_probs = torch.softmax(pred_i / temperature, dim=0)
            
            # Calculate weighted return (expectation)
            weighted_return = torch.sum(top5_probs * actual_i)
            
            # Loss is negative return (we want to maximize return)
            loss_i = -weighted_return
            total_loss += loss_i
        
        return total_loss / batch_size
    
    # Train the model on ALL 60 months (no validation split)
    logger.info("="*80)
    logger.info("TRAINING ON FULL 60 MONTHS")
    logger.info("="*80)
    
    model.train()
    best_loss = float('inf')
    patience_counter = 0
    training_losses = []
    
    # Create data loader
    from torch.utils.data import DataLoader, TensorDataset
    dataset = TensorDataset(train_X_tensor, train_y_tensor)
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
    
    for epoch in range(config['n_epochs']):
        epoch_loss = 0.0
        batch_count = 0
        
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            
            predictions = model(batch_X)
            loss = top5_return_loss(predictions, batch_y)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            batch_count += 1
        
        avg_epoch_loss = epoch_loss / batch_count
        training_losses.append(avg_epoch_loss)
        
        # Early stopping based on training loss plateau
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            patience_counter = 0
            # Save best model state
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        if epoch % 10 == 0 or epoch < 5:
            logger.info(f"Epoch {epoch+1:3d}/100 - Loss: {avg_epoch_loss:.6f} - Best: {best_loss:.6f}")
        
        if patience_counter >= config['early_stopping_patience']:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break
    
    # Restore best model
    model.load_state_dict(best_model_state)
    logger.info(f"Training completed. Best loss: {best_loss:.6f}")
    
    # Make prediction for target month
    logger.info("="*80)
    logger.info("MAKING PREDICTION FOR TARGET MONTH")
    logger.info("="*80)
    
    model.eval()
    with torch.no_grad():
        predictions = model(target_X_tensor)
        
        # Get top 5 factors
        pred_values = predictions[0].cpu().numpy()
        actual_values = target_y_tensor[0].cpu().numpy()
        
        # Get top 5 predicted factors
        top5_indices = np.argsort(pred_values)[-5:][::-1]  # Top 5 in descending order
        
        logger.info(f"TOP 5 PREDICTED FACTORS for {target_date}:")
        for i, idx in enumerate(top5_indices):
            factor_name = factor_names[idx]
            predicted_value = pred_values[idx]
            actual_return = actual_values[idx]
            logger.info(f"  {i+1}. {factor_name}: pred={predicted_value:.4f}, actual={actual_return:.4f}")
        
        # Calculate portfolio return
        top5_actual_returns = actual_values[top5_indices]
        portfolio_return = np.mean(top5_actual_returns)
        
        # Calculate hit rate (how many of predicted top 5 are in actual top 5)
        actual_top5_indices = set(np.argsort(actual_values)[-5:])
        predicted_top5_indices = set(top5_indices)
        hits = len(actual_top5_indices.intersection(predicted_top5_indices))
        hit_rate = hits / 5
        
        logger.info("="*80)
        logger.info("RESULTS")
        logger.info("="*80)
        logger.info(f"Portfolio return (avg of top 5): {portfolio_return:.4f} ({portfolio_return*100:.2f}%)")
        logger.info(f"Hit rate: {hit_rate:.3f} ({hit_rate*100:.1f}%)")
        logger.info(f"Hits: {hits}/5")
        logger.info(f"Individual top-5 returns: {[f'{r:.3f}' for r in top5_actual_returns]}")
        
        # Compare with benchmarks
        equal_weighted_return = np.mean(actual_values)
        actual_top5_return = np.mean(actual_values[list(actual_top5_indices)])
        
        logger.info("="*80)
        logger.info("BENCHMARK COMPARISON")
        logger.info("="*80)
        logger.info(f"Equal-weighted return: {equal_weighted_return:.4f} ({equal_weighted_return*100:.2f}%)")
        logger.info(f"Actual top-5 return: {actual_top5_return:.4f} ({actual_top5_return*100:.2f}%)")
        logger.info(f"Model top-5 return: {portfolio_return:.4f} ({portfolio_return*100:.2f}%)")
        logger.info(f"Outperformance vs EW: {(portfolio_return - equal_weighted_return)*100:.2f} pp")
        logger.info(f"Capture of actual top-5: {(portfolio_return / actual_top5_return)*100:.1f}%")
        
        return {
            'target_date': target_date,
            'portfolio_return': portfolio_return,
            'hit_rate': hit_rate,
            'top5_factors': [factor_names[i] for i in top5_indices],
            'top5_predictions': pred_values[top5_indices],
            'top5_actual_returns': top5_actual_returns,
            'equal_weighted_return': equal_weighted_return,
            'actual_top5_return': actual_top5_return
        }

if __name__ == "__main__":
    results = run_single_window(window_index=119)  # 2015-01-01