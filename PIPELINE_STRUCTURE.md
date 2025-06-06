# T2-NN Clean Pipeline Structure

## 📁 Core Files (25 total)

### 🚀 **Execution Scripts**
- `main.py` - Complete pipeline execution
- `generate_reports.py` - Professional PDF/Excel report generation

### 🔧 **Source Code** (`src/`)
- `__init__.py` - Package initialization
- `config.py` - Configuration and hyperparameters  
- `data.py` - Data loading and preprocessing
- `model.py` - Neural network architecture
- `train.py` - Training procedures and utilities
- `tune.py` - Hyperparameter optimization
- `backtest.py` - Rolling window backtesting
- `evaluate.py` - Performance evaluation metrics

### 📊 **Final Outputs** (`outputs/`)
- `T60_Enhanced_NN.xlsx` - Enhanced factor forecasts (main deliverable)
- `Performance_Report.pdf` - Executive summary report
- `Performance_Report.xlsx` - Comprehensive analysis workbook
- `backtest_results.csv` - Monthly backtest data
- `backtest_analysis.csv` - Performance summary statistics
- `hyperparam_results.csv` - Optimization results

### 📈 **Visualization** (`outputs/plots/`)
- `cumulative_returns.png` - Portfolio performance over time
- `drawdowns.png` - Risk analysis chart
- `hit_rate.png` - Model accuracy over time
- `returns_distribution.png` - Monthly returns histogram
- `returns_heatmap.png` - Performance by year/month
- `rolling_sharpe.png` - Rolling risk-adjusted returns

### 📋 **Documentation**
- `README.md` - Comprehensive documentation
- `.gitignore` - Git ignore patterns
- `requirements.txt` - Python dependencies

## 🧹 **Removed (31 files)**
- All `test_*.py` files (6 files)
- Experimental scripts (`fix_temporal_leakage.py`, `quick_fix_test.py`, etc.)
- Debug utilities (`inspect_training_data.py`, `find_window.py`, etc.)
- Temporary outputs (logs, checkpoints, analysis files)
- Development artifacts and redundant files

## 🎯 **Usage Pipeline**

```bash
# Complete system execution
python main.py                 # Runs hyperparameter tuning + backtest
python generate_reports.py     # Generates professional reports

# Individual components (if needed)
python -c "from src.tune import main; main()"        # Hyperparameter tuning only
python -c "from src.backtest import run_backtest; run_backtest()"  # Backtest only
```

## ✅ **Clean Repository Benefits**

1. **Production Ready** - Only essential pipeline files
2. **Easy Navigation** - Clear structure without clutter  
3. **Reduced Size** - 44% fewer files (25 vs 56)
4. **Professional** - Clean for collaboration and deployment
5. **Focused** - Core functionality only, no experimental code

The repository now contains only the essential components needed to run the complete T2 Neural Network Factor Selection pipeline.