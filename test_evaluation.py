#!/usr/bin/env python3
"""
Test script for evaluation functionality.
Tests the evaluation metrics and comparison functions.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import logging
from src.config import setup_logging
from src.evaluate import (
    calculate_metrics, 
    calculate_benchmark_metrics, 
    compare_performance, 
    calculate_rolling_metrics,
    create_performance_summary,
    save_evaluation_results
)

def create_test_data(n_samples: int = 50, n_factors: int = 83, seed: int = 42):
    """Create synthetic test data for evaluation."""
    np.random.seed(seed)
    
    # Create synthetic predictions with some signal
    predictions = np.random.randn(n_samples, n_factors) * 0.1
    
    # Create actual returns with some correlation to predictions
    noise = np.random.randn(n_samples, n_factors) * 0.05
    actual_returns = predictions * 0.3 + noise  # 30% signal, 70% noise
    
    # Add some stronger signals for top factors to make it more realistic
    for i in range(n_samples):
        # Make some predicted top factors actually perform better
        top_pred_indices = np.argsort(predictions[i])[-10:]  # Top 10 predicted
        actual_returns[i, top_pred_indices] += np.random.randn(10) * 0.01 + 0.005  # Small boost
    
    return predictions, actual_returns

def test_calculate_metrics():
    """Test the calculate_metrics function."""
    logger = logging.getLogger(__name__)
    logger.info("Testing calculate_metrics function...")
    
    # Test with multiple samples
    predictions, actual_returns = create_test_data(n_samples=50, n_factors=83)
    
    metrics = calculate_metrics(predictions, actual_returns, top_k=5)
    
    logger.info("Multi-sample metrics:")
    for key, value in metrics.items():
        if not isinstance(value, list):
            logger.info(f"  {key}: {value}")
    
    # Test with single sample
    single_pred = predictions[0]
    single_actual = actual_returns[0]
    
    single_metrics = calculate_metrics(single_pred, single_actual, top_k=5)
    
    logger.info("Single-sample metrics:")
    for key, value in single_metrics.items():
        if not isinstance(value, list):
            logger.info(f"  {key}: {value}")
    
    # Test with torch tensors
    pred_tensor = torch.FloatTensor(predictions)
    actual_tensor = torch.FloatTensor(actual_returns)
    
    tensor_metrics = calculate_metrics(pred_tensor, actual_tensor, top_k=5)
    
    # Verify results are reasonable
    assert 0 <= metrics['hit_rate'] <= 1, f"Hit rate should be between 0 and 1, got {metrics['hit_rate']}"
    assert -1 <= metrics['spearman_correlation'] <= 1, f"Spearman correlation should be between -1 and 1"
    assert metrics['n_samples'] == 50, f"Should have 50 samples, got {metrics['n_samples']}"
    assert metrics['top_k'] == 5, f"Should have top_k=5, got {metrics['top_k']}"
    assert len(metrics['monthly_returns']) == 50, f"Should have 50 monthly returns"
    
    logger.info("✅ calculate_metrics test passed!")
    return True

def test_benchmark_metrics():
    """Test the benchmark metrics calculation."""
    logger = logging.getLogger(__name__)
    logger.info("Testing calculate_benchmark_metrics function...")
    
    _, actual_returns = create_test_data(n_samples=50, n_factors=83)
    
    benchmark_metrics = calculate_benchmark_metrics(actual_returns, top_k=5)
    
    logger.info("Benchmark metrics:")
    for key, value in benchmark_metrics.items():
        if not isinstance(value, list):
            logger.info(f"  {key}: {value}")
    
    # Verify benchmark results
    assert 'equal_weighted_return' in benchmark_metrics
    assert 'random_top5_return' in benchmark_metrics
    assert len(benchmark_metrics['ew_monthly_returns']) == 50
    assert len(benchmark_metrics['random_monthly_returns']) == 50
    
    logger.info("✅ benchmark_metrics test passed!")
    return True

def test_performance_comparison():
    """Test the performance comparison functionality."""
    logger = logging.getLogger(__name__)
    logger.info("Testing performance comparison...")
    
    predictions, actual_returns = create_test_data(n_samples=50, n_factors=83)
    
    # Calculate all metrics
    model_metrics = calculate_metrics(predictions, actual_returns, top_k=5)
    benchmark_metrics = calculate_benchmark_metrics(actual_returns, top_k=5)
    comparison_metrics = compare_performance(model_metrics, benchmark_metrics)
    
    logger.info("Performance comparison metrics:")
    for key, value in comparison_metrics.items():
        logger.info(f"  {key}: {value}")
    
    # Verify comparison results
    assert 'excess_return_vs_equal_weighted' in comparison_metrics
    assert 'annual_excess_return_vs_ew' in comparison_metrics
    assert 'is_significant_vs_ew' in comparison_metrics
    assert isinstance(comparison_metrics['is_significant_vs_ew'], bool)
    
    logger.info("✅ performance_comparison test passed!")
    return True

def test_rolling_metrics():
    """Test rolling metrics calculation."""
    logger = logging.getLogger(__name__)
    logger.info("Testing rolling metrics calculation...")
    
    # Create a longer time series
    predictions, actual_returns = create_test_data(n_samples=100, n_factors=83)
    model_metrics = calculate_metrics(predictions, actual_returns, top_k=5)
    
    rolling_metrics = calculate_rolling_metrics(model_metrics['monthly_returns'], window_size=12)
    
    logger.info(f"Rolling metrics calculated for {len(rolling_metrics['rolling_returns'])} windows")
    
    # Verify rolling metrics
    expected_windows = len(model_metrics['monthly_returns']) - 12 + 1
    assert len(rolling_metrics['rolling_returns']) == expected_windows
    assert len(rolling_metrics['rolling_sharpe']) == expected_windows
    
    # Test with insufficient data
    short_rolling = calculate_rolling_metrics([0.01, 0.02], window_size=12)
    assert len(short_rolling['rolling_returns']) == 0
    
    logger.info("✅ rolling_metrics test passed!")
    return True

def test_performance_summary():
    """Test performance summary generation."""
    logger = logging.getLogger(__name__)
    logger.info("Testing performance summary generation...")
    
    predictions, actual_returns = create_test_data(n_samples=50, n_factors=83)
    
    # Calculate all metrics
    model_metrics = calculate_metrics(predictions, actual_returns, top_k=5)
    benchmark_metrics = calculate_benchmark_metrics(actual_returns, top_k=5)
    comparison_metrics = compare_performance(model_metrics, benchmark_metrics)
    
    # Generate summary
    summary = create_performance_summary(model_metrics, benchmark_metrics, comparison_metrics)
    
    logger.info("Generated performance summary:")
    logger.info("\n" + summary)
    
    # Verify summary contains key sections
    assert "MODEL PERFORMANCE:" in summary
    assert "BENCHMARK PERFORMANCE:" in summary
    assert "SUCCESS CRITERIA CHECK:" in summary
    assert "Sample Size:" in summary
    
    logger.info("✅ performance_summary test passed!")
    return True

def test_save_results():
    """Test saving evaluation results."""
    logger = logging.getLogger(__name__)
    logger.info("Testing save evaluation results...")
    
    predictions, actual_returns = create_test_data(n_samples=30, n_factors=83)
    
    # Calculate all metrics
    model_metrics = calculate_metrics(predictions, actual_returns, top_k=5)
    benchmark_metrics = calculate_benchmark_metrics(actual_returns, top_k=5)
    comparison_metrics = compare_performance(model_metrics, benchmark_metrics)
    
    # Save results
    summary_path = save_evaluation_results(
        model_metrics, benchmark_metrics, comparison_metrics, 
        output_dir='outputs/test_evaluation'
    )
    
    # Verify files were created
    assert os.path.exists(summary_path)
    assert os.path.exists('outputs/test_evaluation/evaluation_metrics.csv')
    
    # Check file contents
    with open(summary_path, 'r') as f:
        content = f.read()
        assert "FACTOR SELECTION MODEL PERFORMANCE SUMMARY" in content
    
    logger.info(f"Results saved to {summary_path}")
    logger.info("✅ save_results test passed!")
    return True

def test_realistic_scenario():
    """Test with a more realistic scenario."""
    logger = logging.getLogger(__name__)
    logger.info("Testing realistic performance scenario...")
    
    # Create data that should meet success criteria
    np.random.seed(123)
    n_samples, n_factors = 60, 83  # 5 years of data
    
    # Create predictions with stronger signal
    predictions = np.random.randn(n_samples, n_factors) * 0.05
    
    # Create returns with better correlation
    actual_returns = predictions * 0.5 + np.random.randn(n_samples, n_factors) * 0.03
    
    # Boost performance of top predicted factors more consistently
    for i in range(n_samples):
        top_indices = np.argsort(predictions[i])[-5:]  # Top 5 predicted
        actual_returns[i, top_indices] += 0.01  # 1% monthly boost
    
    # Calculate comprehensive evaluation
    model_metrics = calculate_metrics(predictions, actual_returns, top_k=5)
    benchmark_metrics = calculate_benchmark_metrics(actual_returns, top_k=5)
    comparison_metrics = compare_performance(model_metrics, benchmark_metrics)
    
    # Generate and save results
    summary = create_performance_summary(model_metrics, benchmark_metrics, comparison_metrics)
    save_evaluation_results(
        model_metrics, benchmark_metrics, comparison_metrics,
        output_dir='outputs/realistic_evaluation'
    )
    
    logger.info("Realistic scenario results:")
    logger.info(f"  Top-5 Return: {model_metrics['top5_return']:.4f} ({model_metrics['top5_return']*100:.2f}%)")
    logger.info(f"  Hit Rate: {model_metrics['hit_rate']:.3f} ({model_metrics['hit_rate']*100:.1f}%)")
    logger.info(f"  Sharpe Ratio: {model_metrics['sharpe_ratio']:.3f}")
    logger.info(f"  Annual Excess vs EW: {comparison_metrics['annual_excess_return_vs_ew']:.1%}")
    
    # Check if it meets success criteria
    meets_return_criteria = comparison_metrics['annual_excess_return_vs_ew'] > 0.10
    meets_hit_rate_criteria = model_metrics['hit_rate'] > 0.30
    meets_sharpe_criteria = model_metrics['sharpe_ratio'] > 1.0
    
    logger.info(f"  Meets return criteria (>10% annually): {meets_return_criteria}")
    logger.info(f"  Meets hit rate criteria (>30%): {meets_hit_rate_criteria}")
    logger.info(f"  Meets Sharpe criteria (>1.0): {meets_sharpe_criteria}")
    
    logger.info("✅ realistic_scenario test passed!")
    return True

def main():
    """Run all evaluation tests."""
    # Setup logging
    logger = setup_logging('outputs/test_evaluation.log')
    logger.info("Starting evaluation system tests")
    
    success = True
    
    # Run all tests
    tests = [
        ("Basic metrics calculation", test_calculate_metrics),
        ("Benchmark metrics", test_benchmark_metrics),
        ("Performance comparison", test_performance_comparison),
        ("Rolling metrics", test_rolling_metrics),
        ("Performance summary", test_performance_summary),
        ("Save results", test_save_results),
        ("Realistic scenario", test_realistic_scenario)
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
        logger.info("\n🎉 All evaluation tests passed!")
        print("✅ Evaluation system tests completed successfully!")
        return True
    else:
        logger.error("\n💥 Some evaluation tests failed!")
        print("❌ Some evaluation tests failed!")
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