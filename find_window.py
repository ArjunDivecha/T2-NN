#!/usr/bin/env python3
"""
Find the window index for a specific date.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from src.data import load_data, create_rolling_windows

# Load data
forecast_df, actual_df, factor_names, dates = load_data()
rolling_windows = create_rolling_windows(forecast_df, actual_df, window_size=60)

target_date = pd.Timestamp('2015-01-01')

print(f"Looking for window with target date: {target_date}")
print()

for i, (train_X, train_y, target_X, target_y, window_target_date) in enumerate(rolling_windows):
    if window_target_date == target_date:
        print(f"Found! Window index {i} has target date {window_target_date}")
        break
    elif i < 10 or abs((window_target_date - target_date).days) < 400:
        print(f"Window {i}: target = {window_target_date}")

print(f"\nFirst window target: {rolling_windows[0][4]}")
print(f"Last window target: {rolling_windows[-1][4]}")