#!/usr/bin/env python3
"""
Standalone script to run hyperparameter tuning.
This script can be executed directly to perform hyperparameter search.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.tune import main

if __name__ == "__main__":
    print("Starting hyperparameter tuning...")
    try:
        results_df, best_params = main()
        print("✅ Hyperparameter tuning completed successfully!")
        print(f"Best parameters: {best_params}")
        print(f"Total combinations tested: {len(results_df)}")
        print(f"Results saved to outputs/hyperparam_results.csv")
    except Exception as e:
        print(f"❌ Hyperparameter tuning failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)