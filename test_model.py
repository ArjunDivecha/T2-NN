#!/usr/bin/env python3
"""Test script for neural network model and loss function"""

import sys
import os
sys.path.append('src')

import torch
import numpy as np
from src.config import setup_logging, get_device, DEFAULT_CONFIG
from src.model import SimpleNN, top5_return_loss, calculate_top5_metrics, create_model, get_optimizer

def test_model_architecture():
    """Test model creation and forward pass"""
    print("\n🏗️ Testing Model Architecture...")
    
    # Create model with default config
    device = get_device()
    print(f"Using device: {device}")
    
    model = SimpleNN(
        input_size=83,
        hidden_sizes=[512, 256],
        dropout_rate=0.2
    )
    model = model.to(device)
    
    # Test forward pass
    batch_size = 32
    test_input = torch.randn(batch_size, 83, device=device)
    
    with torch.no_grad():
        output = model(test_input)
    
    print(f"✅ Model created successfully")
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Parameters: {model.count_parameters():,}")
    
    assert output.shape == (batch_size, 83), f"Expected shape ({batch_size}, 83), got {output.shape}"
    print("✅ Forward pass test passed")
    
    return model, device

def test_loss_function():
    """Test custom top5 loss function"""
    print("\n🎯 Testing Loss Function...")
    
    device = get_device()
    batch_size = 16
    n_factors = 83
    
    # Create sample data
    # Predictions: higher values should be selected as top 5
    predictions = torch.randn(batch_size, n_factors, device=device)
    
    # Actual returns: create some with known high/low returns
    actual_returns = torch.randn(batch_size, n_factors, device=device)
    
    # Test loss calculation
    loss = top5_return_loss(predictions, actual_returns)
    print(f"✅ Loss computed: {loss.item():.6f}")
    
    # Test that loss is differentiable
    predictions.requires_grad_(True)
    loss = top5_return_loss(predictions, actual_returns)
    loss.backward()
    
    assert predictions.grad is not None, "Gradients not computed"
    print(f"✅ Gradient computation successful")
    print(f"Gradient norm: {predictions.grad.norm().item():.6f}")
    
    return loss

def test_metrics():
    """Test metric calculation"""
    print("\n📊 Testing Metrics...")
    
    device = get_device()
    batch_size = 8
    n_factors = 83
    
    # Create test data where we know the top 5
    predictions = torch.zeros(batch_size, n_factors, device=device)
    actual_returns = torch.zeros(batch_size, n_factors, device=device)
    
    # Make first 5 factors have highest predictions and returns (perfect overlap)
    predictions[:, :5] = torch.randn(batch_size, 5, device=device) + 5  # Very high values
    actual_returns[:, :5] = torch.randn(batch_size, 5, device=device) + 3  # High returns
    
    # Make remaining factors much lower
    predictions[:, 5:] = torch.randn(batch_size, n_factors-5, device=device) - 2
    actual_returns[:, 5:] = torch.randn(batch_size, n_factors-5, device=device) - 2
    
    metrics = calculate_top5_metrics(predictions, actual_returns)
    
    print(f"✅ Metrics calculated:")
    print(f"Average top-5 return: {metrics['avg_top5_return']:.6f}")
    print(f"Hit rate: {metrics['hit_rate']:.2%}")
    
    # Hit rate should be decent since we designed overlap (but not necessarily perfect)
    assert metrics['hit_rate'] > 0.6, f"Expected decent hit rate, got {metrics['hit_rate']}"
    print("✅ Metrics test passed")
    
    return metrics

def test_device_compatibility():
    """Test MPS compatibility on Apple Silicon"""
    print("\n🖥️ Testing Device Compatibility...")
    
    device = get_device()
    print(f"Selected device: {device}")
    
    if device.type == 'mps':
        print("✅ MPS (Metal Performance Shaders) available")
        
        # Test MPS operations
        x = torch.randn(100, 83, device=device)
        y = torch.randn(100, 83, device=device)
        
        # Basic operations
        z = x + y
        loss = (z ** 2).mean()
        
        print("✅ Basic MPS operations successful")
        
    elif device.type == 'cpu':
        print("⚠️ Using CPU (MPS not available)")
        
    return device

def test_end_to_end():
    """Test complete model training step"""
    print("\n🔄 Testing End-to-End Training Step...")
    
    device = get_device()
    
    # Create model and optimizer
    config = DEFAULT_CONFIG.copy()
    config['input_size'] = 83
    
    model = create_model(config, device)
    optimizer = get_optimizer(model, config)
    
    # Sample batch
    batch_size = 16
    X = torch.randn(batch_size, 83, device=device)
    y = torch.randn(batch_size, 83, device=device)
    
    # Training step
    model.train()
    optimizer.zero_grad()
    
    predictions = model(X)
    loss = top5_return_loss(predictions, y)
    loss.backward()
    optimizer.step()
    
    print(f"✅ Training step completed")
    print(f"Loss: {loss.item():.6f}")
    
    # Calculate metrics
    model.eval()
    with torch.no_grad():
        eval_predictions = model(X)
        metrics = calculate_top5_metrics(eval_predictions, y)
    
    print(f"Avg top-5 return: {metrics['avg_top5_return']:.6f}")
    print(f"Hit rate: {metrics['hit_rate']:.2%}")
    
    print("✅ End-to-end test passed")

def main():
    # Setup logging
    logger = setup_logging()
    logger.info("Testing neural network model...")
    
    try:
        # Run all tests
        model, device = test_model_architecture()
        loss = test_loss_function()
        metrics = test_metrics()
        device = test_device_compatibility()
        test_end_to_end()
        
        print(f"\n🎉 All model tests passed!")
        print(f"Model ready for training on {device}")
        
    except Exception as e:
        logger.error(f"Model test failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()