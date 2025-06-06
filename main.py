#!/usr/bin/env python3
"""
Main script for Top-5 Factor Return Optimization.
Runs hyperparameter tuning, training, and backtest with forecast generation.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.config import setup_logging
from src.tune import run_hyperparameter_search
from src.backtest import run_backtest

def main():
    """
    Run the complete Top-5 Factor Optimization pipeline:
    1. Hyperparameter tuning (optional - can skip if already done)
    2. Full rolling window backtest with monthly forecast generation
    """
    
    # Setup logging
    setup_logging('outputs/main.log')
    
    print("🚀 Starting Top-5 Factor Return Optimization")
    print("=" * 60)
    
    # Check if hyperparameter results exist
    hyperparam_file = 'outputs/hyperparam_results.csv'
    
    if not os.path.exists(hyperparam_file):
        print("Step 1: Running hyperparameter tuning...")
        try:
            from src.tune import main as tune_main
            results_df, best_params = tune_main()
            print(f"✅ Hyperparameter tuning completed!")
            print(f"   Best parameters: {best_params}")
            print(f"   Results saved to {hyperparam_file}")
        except Exception as e:
            print(f"❌ Hyperparameter tuning failed: {e}")
            return False
    else:
        print("✅ Using existing hyperparameter results")
    
    print("\nStep 2: Running full backtest with forecast generation...")
    try:
        results_df, forecasts_list = run_backtest(save_forecasts=True, parallel=False)
        
        # Print summary
        valid_results = results_df.dropna()
        if len(valid_results) > 0:
            avg_return = valid_results['portfolio_return'].mean()
            avg_hit_rate = valid_results['hit_rate'].mean()
            sharpe = valid_results['portfolio_return'].mean() / valid_results['portfolio_return'].std()
            
            print(f"✅ Backtest completed successfully!")
            print(f"   Average Monthly Return: {avg_return:.4f} ({avg_return*12:.2f}% annualized)")
            print(f"   Average Hit Rate: {avg_hit_rate:.3f}")
            print(f"   Sharpe Ratio: {sharpe:.3f}")
            print(f"   Total Months: {len(valid_results)}")
        
        print(f"\n📊 Outputs generated:")
        print(f"   • Monthly Forecasts: outputs/T60_Enhanced_NN.xlsx")
        print(f"   • Backtest Results: outputs/backtest_results.csv")
        print(f"   • Hyperparameter Results: outputs/hyperparam_results.csv")
        
        return True
        
    except Exception as e:
        print(f"❌ Backtest failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)