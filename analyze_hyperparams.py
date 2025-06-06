import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ast import literal_eval

# Load the data
df = pd.read_csv('/Users/macbook2024/Dropbox/AAA Backup/A Working/T2 Claude Code Transformer/top5_factor_optimization/outputs/hyperparam_results.csv')

print("=== HYPERPARAMETER TUNING ANALYSIS ===")
print(f"Total experiments: {len(df)}")
print(f"Columns: {list(df.columns)}")
print("\n")

# Parse hidden_sizes column
df['hidden_sizes_parsed'] = df['hidden_sizes'].apply(literal_eval)
df['num_hidden_layers'] = df['hidden_sizes_parsed'].apply(len)
df['first_layer_size'] = df['hidden_sizes_parsed'].apply(lambda x: x[0])
df['model_complexity'] = df['hidden_sizes_parsed'].apply(lambda x: sum(x))

print("=== DATA OVERVIEW ===")
print(df.describe())
print("\n")

print("=== TOP PERFORMERS BY RETURN ===")
top_returns = df.nlargest(5, 'avg_top5_return')[['hidden_sizes', 'learning_rate', 'dropout_rate', 'batch_size', 'weight_decay', 'avg_top5_return', 'avg_hit_rate']]
print(top_returns)
print("\n")

print("=== TOP PERFORMERS BY HIT RATE ===")
top_hit_rates = df.nlargest(5, 'avg_hit_rate')[['hidden_sizes', 'learning_rate', 'dropout_rate', 'batch_size', 'weight_decay', 'avg_top5_return', 'avg_hit_rate']]
print(top_hit_rates)
print("\n")

print("=== PARAMETER IMPACT ANALYSIS ===")

# Learning rate analysis
print("LEARNING RATE IMPACT:")
lr_analysis = df.groupby('learning_rate').agg({
    'avg_top5_return': ['mean', 'std', 'count'],
    'avg_hit_rate': ['mean', 'std'],
    'success_rate': ['mean']
}).round(4)
print(lr_analysis)
print()

# Dropout analysis
print("DROPOUT RATE IMPACT:")
dropout_analysis = df.groupby('dropout_rate').agg({
    'avg_top5_return': ['mean', 'std', 'count'],
    'avg_hit_rate': ['mean', 'std'],
    'success_rate': ['mean']
}).round(4)
print(dropout_analysis)
print()

# Architecture analysis
print("ARCHITECTURE IMPACT:")
arch_analysis = df.groupby('hidden_sizes').agg({
    'avg_top5_return': ['mean', 'std', 'count'],
    'avg_hit_rate': ['mean', 'std'],
    'success_rate': ['mean'],
    'avg_train_time': ['mean']
}).round(4)
print(arch_analysis)
print()

# Weight decay analysis
print("WEIGHT DECAY IMPACT:")
wd_analysis = df.groupby('weight_decay').agg({
    'avg_top5_return': ['mean', 'std', 'count'],
    'avg_hit_rate': ['mean', 'std'],
    'success_rate': ['mean']
}).round(4)
print(wd_analysis)
print()

print("=== OVERFITTING/UNDERFITTING ANALYSIS ===")

# High variance in returns could indicate overfitting
print("Models sorted by return variance (high variance may indicate overfitting):")
variance_analysis = df[['hidden_sizes', 'learning_rate', 'dropout_rate', 'return_std', 'hit_rate_std', 'avg_top5_return', 'success_rate']].sort_values('return_std', ascending=False)
print(variance_analysis)
print()

# Success rate analysis (models that failed to converge)
print("Success rate analysis:")
print(f"Models with 100% success rate: {len(df[df['success_rate'] == 1.0])}")
print(f"Models with <100% success rate: {len(df[df['success_rate'] < 1.0])}")
if len(df[df['success_rate'] < 1.0]) > 0:
    print("Failed models:")
    print(df[df['success_rate'] < 1.0][['hidden_sizes', 'learning_rate', 'dropout_rate', 'success_rate']])
print()

print("=== COMPLEXITY vs PERFORMANCE ===")
complexity_perf = df[['model_complexity', 'avg_top5_return', 'avg_hit_rate', 'avg_train_time']].corr()
print("Correlation matrix (complexity vs performance):")
print(complexity_perf.round(4))
print()

print("=== TRAINING EFFICIENCY ANALYSIS ===")
print("Average training time by architecture:")
time_analysis = df.groupby('hidden_sizes')['avg_train_time'].agg(['mean', 'min', 'max']).round(4)
print(time_analysis)
print()

print("Average epochs to convergence:")
epochs_analysis = df.groupby('hidden_sizes')['avg_epochs'].agg(['mean', 'min', 'max']).round(2)
print(epochs_analysis)
print()

print("=== RETURN vs HIT RATE TRADEOFF ===")
print("Correlation between returns and hit rate:", df['avg_top5_return'].corr(df['avg_hit_rate']).round(4))
print()

# Best balanced performers
df['balanced_score'] = (df['avg_top5_return'] * 0.7 + df['avg_hit_rate'] * 10 * 0.3)  # Weight returns more heavily
print("TOP 5 BALANCED PERFORMERS (70% return weight, 30% hit rate weight):")
balanced_top = df.nlargest(5, 'balanced_score')[['hidden_sizes', 'learning_rate', 'dropout_rate', 'avg_top5_return', 'avg_hit_rate', 'balanced_score']]
print(balanced_top)
print()

print("=== RECOMMENDATIONS ===")

# Find the best overall configuration
best_return_idx = df['avg_top5_return'].idxmax()
best_config = df.loc[best_return_idx]

print("OPTIMAL CONFIGURATION (highest return):")
print(f"Hidden sizes: {best_config['hidden_sizes']}")
print(f"Learning rate: {best_config['learning_rate']}")
print(f"Dropout rate: {best_config['dropout_rate']}")
print(f"Batch size: {best_config['batch_size']}")
print(f"Weight decay: {best_config['weight_decay']}")
print(f"Average return: {best_config['avg_top5_return']:.4f}")
print(f"Average hit rate: {best_config['avg_hit_rate']:.4f}")
print(f"Return std: {best_config['return_std']:.4f}")
print(f"Success rate: {best_config['success_rate']:.2f}")
print()

# Statistical significance check
print("=== STATISTICAL ROBUSTNESS ===")
print("Models with low return standard deviation (more stable):")
stable_models = df[df['return_std'] < df['return_std'].median()].sort_values('avg_top5_return', ascending=False)
print(stable_models[['hidden_sizes', 'learning_rate', 'dropout_rate', 'avg_top5_return', 'return_std']].head())