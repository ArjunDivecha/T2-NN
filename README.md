# T2 Neural Network Factor Selection

A PyTorch-based neural network system for Top-5 Factor Return Optimization. The system learns to identify which 5 factors from 83 available factors will deliver the highest returns, using historical forecast and actual return data.

## 🎯 Overview

This system implements a feed-forward neural network that predicts factor returns and selects the top 5 performing factors each month. The model is trained using a custom loss function that maximizes the average return of the top 5 predicted factors.

## 🏗️ Architecture

- **Model**: Feed-Forward Neural Network (83 → 512 → 256 → 83)
- **Parameters**: 195,667 trainable parameters
- **Activation**: ReLU with 40% dropout
- **Loss Function**: Custom Top-5 Return Maximization
- **Optimizer**: Adam (lr=0.001, weight_decay=1e-5)
- **Training**: 60-month rolling windows with early stopping

## 📊 Performance Summary

- **Analysis Period**: February 2005 - March 2025 (242 months)
- **Annualized Return**: 1.90%
- **Sharpe Ratio**: 0.499
- **Maximum Drawdown**: -10.08%
- **Hit Rate**: 7.8% (overlap between predicted and actual top 5)
- **Win Rate**: 56.2%

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/ArjunDivecha/T2-NN.git
cd T2-NN

# Install dependencies
pip install -r requirements.txt
```

### Usage

```bash
# Run complete system (hyperparameter tuning + backtest + reports)
python main.py

# Run individual components
python src/tune.py          # Hyperparameter tuning only
python src/backtest.py      # Full rolling window backtest
python generate_reports.py  # Generate performance reports

# Run single window analysis
python show_all_predictions.py  # Detailed factor analysis for single month
```

## 📁 Project Structure

```
T2-NN/
├── src/                     # Core modules
│   ├── config.py           # Configuration and hyperparameters
│   ├── data.py             # Data loading and preprocessing
│   ├── model.py            # Neural network architecture
│   ├── train.py            # Training procedures
│   ├── backtest.py         # Rolling window backtesting
│   └── evaluate.py         # Performance evaluation
├── data/                   # Input data files
│   ├── T60.xlsx           # Factor forecasts (83 factors)
│   └── T2_Optimizer.xlsx  # Actual returns
├── outputs/                # Results and reports
│   ├── T60_Enhanced_NN.xlsx        # Enhanced forecasts
│   ├── Performance_Report.pdf     # Executive summary
│   ├── Performance_Report.xlsx    # Detailed analysis
│   └── plots/             # Visualization charts
├── main.py                # Main execution script
├── generate_reports.py    # Report generation
└── requirements.txt       # Python dependencies
```

## 📈 Key Features

### Custom Loss Function
- **Top-5 Return Loss**: Optimizes for the average return of the top 5 predicted factors
- **Temperature-scaled softmax**: Smooth selection mechanism for differentiable training
- **Rolling window validation**: Prevents temporal leakage

### Robust Backtesting
- **242 monthly predictions**: Feb 2005 - Mar 2025
- **60-month training windows**: Uses all available historical data
- **Out-of-time validation**: Strict temporal ordering prevents look-ahead bias
- **Comprehensive metrics**: Return, risk, and attribution analysis

### Professional Reporting
- **PDF Executive Summary**: Clean, professional performance report
- **Excel Workbook**: Detailed analysis with multiple sheets
- **Visualization Charts**: Performance, risk, and factor attribution plots
- **Factor Analysis**: Selection frequency and contribution analysis

## 🔧 Technical Details

### Data Requirements
- **Input Format**: Excel files with 83 factors and monthly observations
- **Date Range**: Minimum 60 months for initial training window
- **Data Validation**: Automatic alignment and missing value checks

### Model Training
- **Early Stopping**: Prevents overfitting with patience mechanism
- **Batch Training**: Configurable batch sizes for memory efficiency
- **Reproducible Results**: Fixed random seeds for consistent outputs
- **Device Support**: MPS (Apple Silicon) and CPU training

### Performance Metrics
- **Return Metrics**: Total, monthly, and annualized returns
- **Risk Metrics**: Volatility, Sharpe ratio, maximum drawdown
- **Model Metrics**: Hit rate, win rate, factor turnover
- **Attribution**: Individual factor contribution analysis

## 📊 Output Files

### Primary Deliverables
- **T60_Enhanced_NN.xlsx**: Enhanced factor forecasts (same format as input)
- **Performance_Report.pdf**: Executive summary with key metrics
- **Performance_Report.xlsx**: Comprehensive analysis workbook

### Supporting Analysis
- **backtest_results.csv**: Monthly backtest details
- **Factor attribution charts**: Selection frequency and performance
- **Risk analysis plots**: Drawdowns, volatility, rolling metrics

## ⚙️ Configuration

Key hyperparameters can be modified in `src/config.py`:

```python
DEFAULT_CONFIG = {
    'hidden_sizes': [512, 256],
    'learning_rate': 0.001,
    'dropout_rate': 0.4,
    'batch_size': 32,
    'weight_decay': 1e-05
}

TRAINING_CONFIG = {
    'n_epochs': 100,
    'early_stopping_patience': 10,
    'random_seed': 42
}
```

## 🔬 Methodology

### Rolling Window Approach
1. **Training**: Use 60 months of historical data
2. **Prediction**: Forecast returns for next month
3. **Selection**: Choose top 5 factors based on predictions
4. **Evaluation**: Compare with actual top 5 performers
5. **Roll Forward**: Move window by 1 month and repeat

### Custom Loss Function
The model uses a differentiable approximation of top-5 selection:
- Temperature-scaled softmax creates smooth factor weights
- Loss maximizes expected return of weighted factor portfolio
- Enables gradient-based optimization of discrete selection problem

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📧 Contact

Arjun Divecha - [@ArjunDivecha](https://github.com/ArjunDivecha)

Project Link: [https://github.com/ArjunDivecha/T2-NN](https://github.com/ArjunDivecha/T2-NN)