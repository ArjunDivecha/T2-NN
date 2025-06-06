import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ast import literal_eval
import warnings
warnings.filterwarnings('ignore')

# Load the data
df = pd.read_csv('/Users/macbook2024/Dropbox/AAA Backup/A Working/T2 Claude Code Transformer/top5_factor_optimization/outputs/hyperparam_results.csv')

# Parse hidden_sizes column
df['hidden_sizes_parsed'] = df['hidden_sizes'].apply(literal_eval)
df['architecture'] = df['hidden_sizes'].apply(lambda x: 'Simple' if '[256]' in x else 'Deep')
df['model_complexity'] = df['hidden_sizes_parsed'].apply(lambda x: sum(x))

# Set up the plotting style
plt.style.use('default')
fig = plt.figure(figsize=(20, 16))

# 1. Return vs Hit Rate Scatter Plot
plt.subplot(3, 4, 1)
colors = {'Simple': 'blue', 'Deep': 'red'}
for arch in df['architecture'].unique():
    mask = df['architecture'] == arch
    plt.scatter(df[mask]['avg_hit_rate'], df[mask]['avg_top5_return'], 
               c=colors[arch], label=arch, alpha=0.7, s=100)
plt.xlabel('Average Hit Rate')
plt.ylabel('Average Top-5 Return')
plt.title('Return vs Hit Rate by Architecture')
plt.legend()
plt.grid(True, alpha=0.3)

# 2. Learning Rate Impact
plt.subplot(3, 4, 2)
lr_groups = df.groupby('learning_rate').agg({
    'avg_top5_return': ['mean', 'std'],
    'avg_hit_rate': ['mean', 'std']
})
x_pos = np.arange(len(lr_groups))
plt.bar(x_pos - 0.2, lr_groups['avg_top5_return']['mean'], 
        width=0.4, label='Return', alpha=0.7, color='blue',
        yerr=lr_groups['avg_top5_return']['std'])
plt.bar(x_pos + 0.2, lr_groups['avg_hit_rate']['mean'] * 10, 
        width=0.4, label='Hit Rate x10', alpha=0.7, color='orange',
        yerr=lr_groups['avg_hit_rate']['std'] * 10)
plt.xlabel('Learning Rate')
plt.ylabel('Performance')
plt.title('Learning Rate Impact')
plt.xticks(x_pos, lr_groups.index)
plt.legend()
plt.grid(True, alpha=0.3)

# 3. Dropout Impact
plt.subplot(3, 4, 3)
dropout_groups = df.groupby('dropout_rate').agg({
    'avg_top5_return': ['mean', 'std'],
    'avg_hit_rate': ['mean', 'std']
})
x_pos = np.arange(len(dropout_groups))
plt.bar(x_pos - 0.2, dropout_groups['avg_top5_return']['mean'], 
        width=0.4, label='Return', alpha=0.7, color='green',
        yerr=dropout_groups['avg_top5_return']['std'])
plt.bar(x_pos + 0.2, dropout_groups['avg_hit_rate']['mean'] * 10, 
        width=0.4, label='Hit Rate x10', alpha=0.7, color='red',
        yerr=dropout_groups['avg_hit_rate']['std'] * 10)
plt.xlabel('Dropout Rate')
plt.ylabel('Performance')
plt.title('Dropout Rate Impact')
plt.xticks(x_pos, dropout_groups.index)
plt.legend()
plt.grid(True, alpha=0.3)

# 4. Architecture Comparison
plt.subplot(3, 4, 4)
arch_data = []
arch_labels = []
for arch in df['architecture'].unique():
    arch_data.append(df[df['architecture'] == arch]['avg_top5_return'].values)
    arch_labels.append(arch)
plt.boxplot(arch_data, labels=arch_labels)
plt.ylabel('Average Top-5 Return')
plt.title('Return Distribution by Architecture')
plt.grid(True, alpha=0.3)

# 5. Training Time vs Performance
plt.subplot(3, 4, 5)
plt.scatter(df['avg_train_time'], df['avg_top5_return'], 
           c=df['model_complexity'], cmap='viridis', s=100, alpha=0.7)
plt.xlabel('Average Training Time (s)')
plt.ylabel('Average Top-5 Return')
plt.title('Training Time vs Return')
plt.colorbar(label='Model Complexity')
plt.grid(True, alpha=0.3)

# 6. Stability Analysis (Return Std vs Mean Return)
plt.subplot(3, 4, 6)
for arch in df['architecture'].unique():
    mask = df['architecture'] == arch
    plt.scatter(df[mask]['return_std'], df[mask]['avg_top5_return'], 
               c=colors[arch], label=arch, alpha=0.7, s=100)
plt.xlabel('Return Standard Deviation')
plt.ylabel('Average Top-5 Return')
plt.title('Stability vs Performance')
plt.legend()
plt.grid(True, alpha=0.3)

# 7. Success Rate Analysis
plt.subplot(3, 4, 7)
success_data = df.groupby('architecture')['success_rate'].mean()
plt.bar(success_data.index, success_data.values, color=['blue', 'red'], alpha=0.7)
plt.ylabel('Success Rate')
plt.title('Model Convergence Success Rate')
plt.ylim(0, 1.1)
for i, v in enumerate(success_data.values):
    plt.text(i, v + 0.02, f'{v:.2f}', ha='center')
plt.grid(True, alpha=0.3)

# 8. Epochs to Convergence
plt.subplot(3, 4, 8)
epochs_data = []
for arch in df['architecture'].unique():
    epochs_data.append(df[df['architecture'] == arch]['avg_epochs'].values)
plt.boxplot(epochs_data, labels=df['architecture'].unique())
plt.ylabel('Average Epochs to Convergence')
plt.title('Training Efficiency by Architecture')
plt.grid(True, alpha=0.3)

# 9. Weight Decay Impact
plt.subplot(3, 4, 9)
wd_groups = df.groupby('weight_decay').agg({
    'avg_top5_return': ['mean', 'std']
})
plt.bar(range(len(wd_groups)), wd_groups['avg_top5_return']['mean'], 
        yerr=wd_groups['avg_top5_return']['std'], alpha=0.7, color='purple')
plt.xlabel('Weight Decay')
plt.ylabel('Average Top-5 Return')
plt.title('Weight Decay Impact')
plt.xticks(range(len(wd_groups)), [f'{x:.0e}' for x in wd_groups.index])
plt.grid(True, alpha=0.3)

# 10. Parameter Correlation Heatmap
plt.subplot(3, 4, 10)
# Create a numerical representation for correlation
df_corr = df[['learning_rate', 'dropout_rate', 'model_complexity', 
              'avg_top5_return', 'avg_hit_rate', 'return_std']].corr()
sns.heatmap(df_corr, annot=True, cmap='coolwarm', center=0, 
            square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
plt.title('Parameter Correlation Matrix')

# 11. Risk-Return Scatter
plt.subplot(3, 4, 11)
# Create risk-adjusted return (return / std)
df['risk_adjusted_return'] = df['avg_top5_return'] / df['return_std']
for arch in df['architecture'].unique():
    mask = df['architecture'] == arch
    plt.scatter(df[mask]['return_std'], df[mask]['avg_top5_return'], 
               c=colors[arch], label=arch, alpha=0.7, s=100)
    
# Add efficient frontier line
plt.xlabel('Return Standard Deviation (Risk)')
plt.ylabel('Average Top-5 Return')
plt.title('Risk-Return Profile')
plt.legend()
plt.grid(True, alpha=0.3)

# 12. Best Configurations Highlight
plt.subplot(3, 4, 12)
# Top 3 by return
top_3 = df.nlargest(3, 'avg_top5_return')
configs = []
returns = []
for _, row in top_3.iterrows():
    config = f"LR:{row['learning_rate']}\nDO:{row['dropout_rate']}\n{row['architecture']}"
    configs.append(config)
    returns.append(row['avg_top5_return'])

plt.bar(range(len(configs)), returns, color=['gold', 'silver', '#CD7F32'], alpha=0.8)
plt.ylabel('Average Top-5 Return')
plt.title('Top 3 Configurations')
plt.xticks(range(len(configs)), [f'#{i+1}' for i in range(len(configs))])
for i, (config, ret) in enumerate(zip(configs, returns)):
    plt.text(i, ret + 0.02, f'{ret:.3f}', ha='center', fontweight='bold')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/macbook2024/Dropbox/AAA Backup/A Working/T2 Claude Code Transformer/top5_factor_optimization/outputs/hyperparameter_analysis.png', 
            dpi=300, bbox_inches='tight')
plt.show()

print("Visualization saved to outputs/hyperparameter_analysis.png")