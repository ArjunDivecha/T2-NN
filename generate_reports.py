#!/usr/bin/env python3
"""
Generate comprehensive performance reports and analysis.
Creates PDF and Excel outputs for the Top-5 Factor Optimization results.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set up plotting
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

def load_results():
    """Load all backtest results and data."""
    print("📊 Loading backtest results...")
    
    # Load backtest results
    results_df = pd.read_csv('outputs/backtest_results.csv')
    results_df['date'] = pd.to_datetime(results_df['date'])
    results_df = results_df.sort_values('date')
    
    # Load analysis
    analysis_df = pd.read_csv('outputs/backtest_analysis.csv')
    
    # Load original data for comparison
    from src.data import load_data
    forecast_df, actual_df, factor_names, dates = load_data()
    
    print(f"✅ Loaded {len(results_df)} monthly results")
    return results_df, analysis_df, factor_names, dates

def calculate_performance_metrics(results_df):
    """Calculate comprehensive performance metrics."""
    print("📈 Calculating performance metrics...")
    
    returns = results_df['portfolio_return'].values
    hit_rates = results_df['hit_rate'].values
    dates = results_df['date'].values
    
    # Basic metrics - returns are in percentage form, convert to decimal
    n_months = len(returns)
    returns_decimal = returns / 100  # Convert from percentage to decimal
    cumulative_returns = (1 + pd.Series(returns_decimal)).cumprod()
    total_return = cumulative_returns.iloc[-1] - 1
    avg_monthly_return = np.mean(returns_decimal)
    monthly_std = np.std(returns_decimal)
    
    # Annualized metrics
    annual_return = avg_monthly_return * 12
    annual_volatility = monthly_std * np.sqrt(12)
    sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0
    
    # Hit rate metrics
    avg_hit_rate = np.mean(hit_rates)
    hit_rate_std = np.std(hit_rates)
    
    # Win/Loss metrics
    winning_months = np.sum(returns_decimal > 0)
    losing_months = np.sum(returns_decimal < 0)
    win_rate = winning_months / n_months
    
    # Risk metrics (use decimal values)
    max_return = np.max(returns_decimal)
    min_return = np.min(returns_decimal)
    
    # Calculate drawdowns (using cumulative returns calculated above)
    running_max = cumulative_returns.expanding().max()
    drawdowns = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdowns.min()
    
    # Monthly performance distribution
    positive_months = returns_decimal[returns_decimal > 0]
    negative_months = returns_decimal[returns_decimal < 0]
    avg_positive_return = np.mean(positive_months) if len(positive_months) > 0 else 0
    avg_negative_return = np.mean(negative_months) if len(negative_months) > 0 else 0
    
    # Rolling metrics (use decimal returns)
    rolling_12m_returns = pd.Series(returns_decimal).rolling(12).mean() * 12
    rolling_12m_vol = pd.Series(returns_decimal).rolling(12).std() * np.sqrt(12)
    rolling_12m_sharpe = rolling_12m_returns / rolling_12m_vol
    
    metrics = {
        'n_months': n_months,
        'total_return': total_return,
        'avg_monthly_return': avg_monthly_return,
        'annual_return': annual_return,
        'monthly_volatility': monthly_std,
        'annual_volatility': annual_volatility,
        'sharpe_ratio': sharpe_ratio,
        'avg_hit_rate': avg_hit_rate,
        'hit_rate_std': hit_rate_std,
        'win_rate': win_rate,
        'winning_months': winning_months,
        'losing_months': losing_months,
        'max_return': max_return,
        'min_return': min_return,
        'max_drawdown': max_drawdown,
        'avg_positive_return': avg_positive_return,
        'avg_negative_return': avg_negative_return,
        'best_month': dates[np.argmax(returns)],
        'worst_month': dates[np.argmin(returns)],
        'rolling_12m_returns': rolling_12m_returns,
        'rolling_12m_sharpe': rolling_12m_sharpe,
        'cumulative_returns': cumulative_returns,
        'drawdowns': drawdowns
    }
    
    print(f"✅ Calculated {len(metrics)} performance metrics")
    return metrics

def analyze_factor_performance(results_df, factor_names):
    """Analyze individual factor performance and attribution."""
    print("🔍 Analyzing factor performance...")
    
    # Count factor selections
    factor_counts = {}
    factor_returns = {}
    
    for i in range(1, 6):  # Top 5 factors
        factor_col = f'top5_factor_{i}'
        return_col = f'top5_return_{i}'
        
        if factor_col in results_df.columns and return_col in results_df.columns:
            factors = results_df[factor_col].values
            returns = results_df[return_col].values
            
            for factor, ret in zip(factors, returns):
                if pd.notna(factor) and pd.notna(ret):
                    if factor not in factor_counts:
                        factor_counts[factor] = 0
                        factor_returns[factor] = []
                    factor_counts[factor] += 1
                    factor_returns[factor].append(ret)
    
    # Calculate factor statistics
    factor_stats = []
    for factor in factor_names:
        if factor in factor_counts:
            returns = factor_returns[factor]
            stats = {
                'factor': factor,
                'selection_count': factor_counts[factor],
                'selection_frequency': factor_counts[factor] / len(results_df),
                'avg_return': np.mean(returns),
                'total_return': np.sum(returns),
                'volatility': np.std(returns),
                'best_return': np.max(returns),
                'worst_return': np.min(returns),
                'win_rate': np.sum(np.array(returns) > 0) / len(returns)
            }
        else:
            stats = {
                'factor': factor,
                'selection_count': 0,
                'selection_frequency': 0,
                'avg_return': 0,
                'total_return': 0,
                'volatility': 0,
                'best_return': 0,
                'worst_return': 0,
                'win_rate': 0
            }
        factor_stats.append(stats)
    
    factor_analysis = pd.DataFrame(factor_stats)
    factor_analysis = factor_analysis.sort_values('selection_count', ascending=False)
    
    print(f"✅ Analyzed {len(factor_names)} factors")
    return factor_analysis

def create_performance_charts(results_df, metrics, save_dir='outputs/plots'):
    """Create comprehensive performance visualization charts."""
    print("📊 Creating performance charts...")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Set up the plotting style
    plt.style.use('default')
    fig_size = (12, 8)
    
    # 1. Cumulative Returns Chart
    plt.figure(figsize=fig_size)
    dates = results_df['date']
    cumulative_returns = metrics['cumulative_returns']
    
    plt.plot(dates, cumulative_returns, linewidth=2, label='Portfolio', color='#2E86AB')
    plt.axhline(y=1, color='gray', linestyle='--', alpha=0.5, label='Breakeven')
    plt.title('Cumulative Returns Over Time', fontsize=14, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/cumulative_returns.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Monthly Returns Distribution
    plt.figure(figsize=fig_size)
    returns_decimal = results_df['portfolio_return'] / 100  # Convert to decimal
    plt.hist(returns_decimal * 100, bins=30, alpha=0.7, color='#A23B72', edgecolor='black')  # Show in %
    plt.axvline(returns_decimal.mean() * 100, color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {returns_decimal.mean()*100:.2f}%')
    plt.title('Monthly Returns Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Monthly Return (%)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/returns_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Drawdown Chart
    plt.figure(figsize=fig_size)
    drawdowns = metrics['drawdowns'] * 100  # Convert to percentage
    plt.fill_between(dates, drawdowns, 0, alpha=0.7, color='#F18F01', label='Drawdown')
    plt.title('Portfolio Drawdowns Over Time', fontsize=14, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Drawdown (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/drawdowns.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Hit Rate Over Time
    plt.figure(figsize=fig_size)
    hit_rates = results_df['hit_rate']
    rolling_hit_rate = hit_rates.rolling(12).mean()
    
    plt.plot(dates, hit_rates, alpha=0.3, color='gray', label='Monthly Hit Rate')
    plt.plot(dates, rolling_hit_rate, linewidth=2, color='#C73E1D', label='12-Month Rolling Average')
    plt.axhline(y=hit_rates.mean(), color='blue', linestyle='--', alpha=0.7, label=f'Overall Average: {hit_rates.mean():.3f}')
    plt.title('Hit Rate Over Time', fontsize=14, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Hit Rate')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/hit_rate.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Rolling 12-Month Sharpe Ratio
    plt.figure(figsize=fig_size)
    rolling_sharpe = metrics['rolling_12m_sharpe']
    valid_sharpe = rolling_sharpe.dropna()
    valid_dates = dates[11:]  # Skip first 11 months
    
    plt.plot(valid_dates, valid_sharpe, linewidth=2, color='#3F784C')
    plt.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    plt.axhline(y=metrics['sharpe_ratio'], color='red', linestyle='--', alpha=0.7, 
                label=f'Overall Sharpe: {metrics["sharpe_ratio"]:.3f}')
    plt.title('Rolling 12-Month Sharpe Ratio', fontsize=14, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Sharpe Ratio')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/rolling_sharpe.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. Monthly Returns Heatmap (by year and month)
    plt.figure(figsize=(14, 8))
    results_df['year'] = results_df['date'].dt.year
    results_df['month'] = results_df['date'].dt.month
    
    # Create pivot table for heatmap
    heatmap_data = results_df.pivot(index='year', columns='month', values='portfolio_return')
    
    # Create heatmap
    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn', center=0,
                cbar_kws={'label': 'Monthly Return'})
    plt.title('Monthly Returns Heatmap', fontsize=14, fontweight='bold')
    plt.xlabel('Month')
    plt.ylabel('Year')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/returns_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Created 6 performance charts in {save_dir}")
    return True

def create_excel_report(results_df, metrics, factor_analysis, output_file='outputs/Performance_Report.xlsx'):
    """Create comprehensive Excel performance report."""
    print("📋 Creating Excel performance report...")
    
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        
        # Sheet 1: Executive Summary
        summary_data = {
            'Metric': [
                'Analysis Period',
                'Total Months',
                'Total Return',
                'Average Monthly Return',
                'Annualized Return',
                'Monthly Volatility', 
                'Annualized Volatility',
                'Sharpe Ratio',
                'Average Hit Rate',
                'Win Rate',
                'Maximum Monthly Return',
                'Minimum Monthly Return',
                'Maximum Drawdown',
                'Best Month',
                'Worst Month'
            ],
            'Value': [
                f"{results_df['date'].min().strftime('%Y-%m')} to {results_df['date'].max().strftime('%Y-%m')}",
                f"{metrics['n_months']} months",
                f"{metrics['total_return']:.4f} ({metrics['total_return']*100:.2f}%)",
                f"{metrics['avg_monthly_return']:.4f} ({metrics['avg_monthly_return']*100:.2f}%)",
                f"{metrics['annual_return']:.4f} ({metrics['annual_return']*100:.2f}%)",
                f"{metrics['monthly_volatility']:.4f} ({metrics['monthly_volatility']*100:.2f}%)",
                f"{metrics['annual_volatility']:.4f} ({metrics['annual_volatility']*100:.2f}%)",
                f"{metrics['sharpe_ratio']:.3f}",
                f"{metrics['avg_hit_rate']:.3f} ({metrics['avg_hit_rate']*100:.1f}%)",
                f"{metrics['win_rate']:.3f} ({metrics['win_rate']*100:.1f}%)",
                f"{metrics['max_return']:.4f} ({metrics['max_return']*100:.2f}%)",
                f"{metrics['min_return']:.4f} ({metrics['min_return']*100:.2f}%)",
                f"{metrics['max_drawdown']:.4f} ({metrics['max_drawdown']*100:.2f}%)",
                pd.Timestamp(metrics['best_month']).strftime('%Y-%m'),
                pd.Timestamp(metrics['worst_month']).strftime('%Y-%m')
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Executive Summary', index=False)
        
        # Sheet 2: Monthly Results
        monthly_results = results_df.copy()
        monthly_results['portfolio_return_pct'] = monthly_results['portfolio_return'] * 100
        monthly_results['hit_rate_pct'] = monthly_results['hit_rate'] * 100
        monthly_results.to_excel(writer, sheet_name='Monthly Results', index=False)
        
        # Sheet 3: Factor Analysis
        factor_analysis['avg_return_pct'] = factor_analysis['avg_return'] * 100
        factor_analysis['selection_frequency_pct'] = factor_analysis['selection_frequency'] * 100
        factor_analysis['win_rate_pct'] = factor_analysis['win_rate'] * 100
        factor_analysis.to_excel(writer, sheet_name='Factor Analysis', index=False)
        
        # Sheet 4: Rolling Performance
        rolling_data = pd.DataFrame({
            'Date': results_df['date'],
            'Monthly_Return': results_df['portfolio_return'],
            'Cumulative_Return': metrics['cumulative_returns'],
            'Drawdown': metrics['drawdowns'],
            'Rolling_12M_Return': metrics['rolling_12m_returns'],
            'Rolling_12M_Sharpe': metrics['rolling_12m_sharpe'],
            'Hit_Rate': results_df['hit_rate']
        })
        rolling_data.to_excel(writer, sheet_name='Rolling Performance', index=False)
        
        # Sheet 5: Top Performers
        top_months = results_df.nlargest(20, 'portfolio_return')[['date', 'portfolio_return', 'hit_rate']].copy()
        top_months['portfolio_return_pct'] = top_months['portfolio_return'] * 100
        top_months.to_excel(writer, sheet_name='Top 20 Months', index=False)
        
        # Sheet 6: Worst Performers  
        worst_months = results_df.nsmallest(20, 'portfolio_return')[['date', 'portfolio_return', 'hit_rate']].copy()
        worst_months['portfolio_return_pct'] = worst_months['portfolio_return'] * 100
        worst_months.to_excel(writer, sheet_name='Worst 20 Months', index=False)
    
    print(f"✅ Excel report saved to {output_file}")
    return True

def generate_pdf_report(results_df, metrics, factor_analysis, output_file='outputs/Performance_Report.pdf'):
    """Generate comprehensive PDF performance report."""
    print("📄 Creating PDF performance report...")
    
    try:
        from matplotlib.backends.backend_pdf import PdfPages
        import matplotlib.pyplot as plt
        
        with PdfPages(output_file) as pdf:
            
            # Page 1: Title and Executive Summary
            fig, ax = plt.subplots(figsize=(8.5, 11))
            ax.axis('off')
            
            # Title section with better styling
            ax.text(0.5, 0.95, 'Top-5 Factor Optimization', 
                   horizontalalignment='center', fontsize=24, fontweight='bold', color='#2C3E50')
            ax.text(0.5, 0.91, 'Performance Report', 
                   horizontalalignment='center', fontsize=18, color='#34495E')
            ax.text(0.5, 0.87, f'Analysis Period: {results_df["date"].min().strftime("%B %Y")} - {results_df["date"].max().strftime("%B %Y")}', 
                   horizontalalignment='center', fontsize=12, color='#7F8C8D')
            ax.text(0.5, 0.84, f'Generated: {datetime.now().strftime("%B %d, %Y at %H:%M")}', 
                   horizontalalignment='center', fontsize=10, color='#95A5A6')
            
            # Calculate actual turnover by comparing top 5 factors month-to-month
            turnover_rates = []
            for i in range(1, len(results_df)):
                current_factors = set()
                previous_factors = set()
                
                # Get current month's top 5 factors
                for j in range(1, 6):
                    factor_col = f'top5_factor_{j}'
                    if factor_col in results_df.columns:
                        current_factor = results_df.iloc[i][factor_col]
                        if pd.notna(current_factor):
                            current_factors.add(current_factor)
                
                # Get previous month's top 5 factors
                for j in range(1, 6):
                    factor_col = f'top5_factor_{j}'
                    if factor_col in results_df.columns:
                        previous_factor = results_df.iloc[i-1][factor_col]
                        if pd.notna(previous_factor):
                            previous_factors.add(previous_factor)
                
                # Calculate turnover as percentage of portfolio that changed
                if len(previous_factors) > 0:
                    unchanged = len(current_factors.intersection(previous_factors))
                    turnover_rate = (5 - unchanged) / 5 * 100  # 5 factors total
                    turnover_rates.append(turnover_rate)
            
            avg_turnover = np.mean(turnover_rates) if turnover_rates else 0
            
            # Create a focused table with exactly the requested metrics
            table_data = [
                ['Model Architecture', ''],
                ['Network Type', 'Feed-Forward Neural Network'],
                ['Architecture', '83 → 512 → 256 → 83 (195,667 params)'],
                ['Activation', 'ReLU + Dropout (40%)'],
                ['Loss Function', 'Custom Top-5 Return Loss'],
                ['Optimizer', 'Adam (lr=0.001, wd=1e-5)'],
                ['Training', '60-Month Rolling, Early Stop'],
                ['', ''],
                ['Performance Metrics', ''],
                ['Annualized Return', f'{metrics["annual_return"]*100:.2f}%'],
                ['Annualized Volatility', f'{metrics["annual_volatility"]*100:.2f}%'],
                ['Sharpe Ratio', f'{metrics["sharpe_ratio"]:.3f}'],
                ['Maximum Drawdown', f'{metrics["max_drawdown"]*100:.2f}%'],
                ['Win Rate', f'{metrics["win_rate"]*100:.1f}%'],
                ['Monthly Turnover', f'{avg_turnover:.1f}%']
            ]
            
            # Draw the table with alternating row colors
            row_height = 0.025
            col_widths = [0.4, 0.3]
            start_y = 0.75
            
            actual_row = 0
            for i, (label, value) in enumerate(table_data):
                # Skip empty rows for spacing but don't increment actual_row
                if label == '' and value == '':
                    continue
                    
                y_pos = start_y - actual_row * row_height
                
                # Header rows (section titles)
                if value == '' and label != '':
                    # Draw header background
                    rect = plt.Rectangle((0.1, y_pos - row_height/2), 0.7, row_height, 
                                       facecolor='#3498DB', alpha=0.8, transform=ax.transAxes)
                    ax.add_patch(rect)
                    ax.text(0.12, y_pos, label, fontsize=12, fontweight='bold', 
                           color='white', verticalalignment='center')
                else:
                    # Alternating row colors for data rows
                    if actual_row % 2 == 0:
                        rect = plt.Rectangle((0.1, y_pos - row_height/2), 0.7, row_height, 
                                           facecolor='#ECF0F1', alpha=0.5, transform=ax.transAxes)
                        ax.add_patch(rect)
                    
                    # Label column
                    ax.text(0.12, y_pos, label, fontsize=11, verticalalignment='center', 
                           color='#2C3E50', fontweight='500')
                    # Value column
                    ax.text(0.75, y_pos, value, fontsize=11, verticalalignment='center', 
                           horizontalalignment='right', color='#27AE60' if 'Return' in label or 'Win' in label 
                           else '#E74C3C' if 'Drawdown' in label else '#2C3E50',
                           fontweight='600' if any(x in label for x in ['Return', 'Sharpe', 'Turnover']) else 'normal')
                
                actual_row += 1
            
            # Add border around the table
            visible_rows = len([x for x in table_data if not (x[0] == '' and x[1] == '')])
            rect = plt.Rectangle((0.1, start_y - visible_rows * row_height + row_height/2), 
                               0.7, visible_rows * row_height, 
                               fill=False, edgecolor='#BDC3C7', linewidth=1.5, transform=ax.transAxes)
            ax.add_patch(rect)
            
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            
            # Page 2: Cumulative Returns Chart
            fig, ax = plt.subplots(figsize=(11, 8.5))
            dates = results_df['date']
            cumulative_returns = metrics['cumulative_returns']
            
            ax.plot(dates, cumulative_returns, linewidth=2, color='#2E86AB')
            ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
            ax.set_title('Cumulative Returns Over Time', fontsize=16, fontweight='bold')
            ax.set_xlabel('Date')
            ax.set_ylabel('Cumulative Return')
            ax.grid(True, alpha=0.3)
            
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            
            # Page 3: Performance Metrics Charts
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(11, 8.5))
            
            # Monthly returns distribution
            returns = results_df['portfolio_return']
            ax1.hist(returns, bins=20, alpha=0.7, color='#A23B72', edgecolor='black')
            ax1.axvline(returns.mean(), color='red', linestyle='--', linewidth=2)
            ax1.set_title('Monthly Returns Distribution')
            ax1.set_xlabel('Monthly Return')
            ax1.set_ylabel('Frequency')
            
            # Drawdown
            drawdowns = metrics['drawdowns'] * 100
            ax2.fill_between(dates, drawdowns, 0, alpha=0.7, color='#F18F01')
            ax2.set_title('Portfolio Drawdowns')
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Drawdown (%)')
            
            # Hit rate over time
            hit_rates = results_df['hit_rate']
            rolling_hit_rate = hit_rates.rolling(12).mean()
            ax3.plot(dates, rolling_hit_rate, linewidth=2, color='#C73E1D')
            ax3.axhline(y=hit_rates.mean(), color='blue', linestyle='--', alpha=0.7)
            ax3.set_title('12-Month Rolling Hit Rate')
            ax3.set_xlabel('Date')
            ax3.set_ylabel('Hit Rate')
            
            # Rolling Sharpe
            rolling_sharpe = metrics['rolling_12m_sharpe'].dropna()
            valid_dates = dates[11:]
            ax4.plot(valid_dates, rolling_sharpe, linewidth=2, color='#3F784C')
            ax4.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
            ax4.set_title('Rolling 12-Month Sharpe Ratio')
            ax4.set_xlabel('Date')
            ax4.set_ylabel('Sharpe Ratio')
            
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            
            # Page 4: Factor Analysis
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8.5))
            
            # Top 15 most selected factors
            top_factors = factor_analysis.head(15)
            ax1.barh(range(len(top_factors)), top_factors['selection_count'])
            ax1.set_yticks(range(len(top_factors)))
            ax1.set_yticklabels(top_factors['factor'], fontsize=8)
            ax1.set_title('Top 15 Most Selected Factors')
            ax1.set_xlabel('Selection Count')
            
            # Top performers by average return
            top_performers = factor_analysis[factor_analysis['selection_count'] > 0].nlargest(15, 'avg_return')
            ax2.barh(range(len(top_performers)), top_performers['avg_return'] * 100)
            ax2.set_yticks(range(len(top_performers)))
            ax2.set_yticklabels(top_performers['factor'], fontsize=8)
            ax2.set_title('Top 15 Factors by Average Return')
            ax2.set_xlabel('Average Return (%)')
            
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
        
        print(f"✅ PDF report saved to {output_file}")
        return True
        
    except ImportError:
        print("❌ PDF generation requires matplotlib. Skipping PDF report.")
        return False

def main():
    """Generate all performance reports and analysis."""
    print("🚀 Starting comprehensive report generation...")
    print("="*60)
    
    # Load data
    results_df, analysis_df, factor_names, dates = load_results()
    
    # Calculate metrics
    metrics = calculate_performance_metrics(results_df)
    
    # Analyze factors
    factor_analysis = analyze_factor_performance(results_df, factor_names)
    
    # Create charts
    create_performance_charts(results_df, metrics)
    
    # Create Excel report
    create_excel_report(results_df, metrics, factor_analysis)
    
    # Create PDF report
    generate_pdf_report(results_df, metrics, factor_analysis)
    
    # Update todo status
    print("\n" + "="*60)
    print("✅ REPORT GENERATION COMPLETE!")
    print("\n📊 Generated outputs:")
    print("   • Performance_Report.xlsx - Comprehensive Excel analysis")
    print("   • Performance_Report.pdf - Executive summary report") 
    print("   • plots/*.png - Performance visualization charts")
    print("\n📈 Key Performance Summary:")
    print(f"   • {metrics['n_months']} months analyzed")
    print(f"   • {metrics['annual_return']:.4f} ({metrics['annual_return']*100:.2f}%) annualized return")
    print(f"   • {metrics['sharpe_ratio']:.3f} Sharpe ratio")
    print(f"   • {metrics['avg_hit_rate']:.3f} ({metrics['avg_hit_rate']*100:.1f}%) average hit rate")
    print(f"   • {metrics['max_drawdown']:.4f} ({metrics['max_drawdown']*100:.2f}%) maximum drawdown")

if __name__ == "__main__":
    main()