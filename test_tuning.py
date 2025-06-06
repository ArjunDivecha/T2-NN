#!/usr/bin/env python3
"""
Test script for hyperparameter tuning functionality.
Tests with a smaller grid and fewer months for quick validation.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import logging
from src.config import setup_logging
from src.tune import generate_hyperparameter_combinations, run_hyperparameter_search, get_best_hyperparameters

# Override hyperparameter grid for testing
TEST_HYPERPARAM_GRID = {
    'hidden_sizes': [
        [256],
        [512, 256]
    ],
    'learning_rate': [1e-3, 1e-2],
    'dropout_rate': [0.0, 0.2],
    'batch_size': [32],
    'weight_decay': [0, 1e-5]
}

def test_hyperparameter_generation():
    """Test hyperparameter combination generation."""
    logger = logging.getLogger(__name__)
    
    # Temporarily replace the grid for testing
    import src.tune
    original_grid = src.tune.HYPERPARAM_GRID
    src.tune.HYPERPARAM_GRID = TEST_HYPERPARAM_GRID
    
    try:
        combinations = generate_hyperparameter_combinations()
        
        logger.info(f"Generated {len(combinations)} test combinations")
        expected_count = 2 * 2 * 2 * 1 * 2  # 16 combinations
        
        if len(combinations) == expected_count:
            logger.info("✅ Hyperparameter generation test passed!")
            
            # Show first few combinations
            for i, combo in enumerate(combinations[:3]):
                logger.info(f"  Example {i+1}: {combo}")
            
            return True
        else:
            logger.error(f"❌ Expected {expected_count} combinations, got {len(combinations)}")
            return False
            
    finally:
        # Restore original grid
        src.tune.HYPERPARAM_GRID = original_grid

def test_hyperparameter_search():
    """Test hyperparameter search with synthetic data and small grid."""
    logger = logging.getLogger(__name__)
    
    # Check if we have real data files
    data_dir = 'data/'
    t60_path = os.path.join(data_dir, 'T60.xlsx')
    t2_path = os.path.join(data_dir, 'T2_Optimizer.xlsx')
    
    if not (os.path.exists(t60_path) and os.path.exists(t2_path)):
        logger.warning("Real data files not found - skipping full search test")
        logger.info("✅ Hyperparameter search test skipped (no data)")
        return True
    
    # Temporarily replace the grid for testing
    import src.tune
    original_grid = src.tune.HYPERPARAM_GRID
    src.tune.HYPERPARAM_GRID = TEST_HYPERPARAM_GRID
    
    try:
        logger.info("Testing hyperparameter search with real data...")
        
        # Run with just 3 months and save results
        results_df = run_hyperparameter_search(
            data_dir=data_dir,
            n_months=3,  # Very small for testing
            save_results=True
        )
        
        logger.info(f"Search completed with {len(results_df)} results")
        
        # Test getting best parameters
        best_params = get_best_hyperparameters(results_df)
        
        logger.info("✅ Hyperparameter search test passed!")
        logger.info(f"Best parameters: {best_params}")
        
        # Show top 3 results
        logger.info("Top 3 results:")
        for i, (_, row) in enumerate(results_df.head(3).iterrows()):
            logger.info(f"  {i+1}. Return: {row['avg_top5_return']:.4f}, "
                       f"Hit Rate: {row['avg_hit_rate']:.3f}, "
                       f"Config: {row['hidden_sizes']}, LR: {row['learning_rate']:.0e}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Hyperparameter search test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Restore original grid
        src.tune.HYPERPARAM_GRID = original_grid

def main():
    """Run all hyperparameter tuning tests."""
    # Setup logging
    logger = setup_logging('outputs/test_tuning.log')
    logger.info("Starting hyperparameter tuning tests")
    
    success = True
    
    # Test 1: Hyperparameter generation
    logger.info("Test 1: Hyperparameter combination generation")
    success &= test_hyperparameter_generation()
    
    # Test 2: Hyperparameter search (if data available)
    logger.info("Test 2: Hyperparameter search")
    success &= test_hyperparameter_search()
    
    if success:
        logger.info("✅ All hyperparameter tuning tests passed!")
        print("✅ Hyperparameter tuning tests completed successfully!")
        return True
    else:
        logger.error("❌ Some hyperparameter tuning tests failed!")
        print("❌ Some hyperparameter tuning tests failed!")
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