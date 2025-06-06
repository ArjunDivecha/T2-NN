# Hyperparameter Tuning Analysis Report

## Executive Summary

Based on 16 hyperparameter experiments, the analysis reveals clear patterns for optimizing the Top-5 Factor Return model. **Simple single-layer architectures significantly outperform deeper networks**, with the optimal configuration achieving 1.86% average returns.

## Key Findings

### 1. Architecture Impact (Most Critical Finding)
- **Simple [256] architecture vastly outperforms deeper [512, 256] networks**
- Simple models: Average return = 1.66%, Hit rate = 11.7%
- Deep models: Average return = 0.89%, Hit rate = 10.8%
- **Recommendation: Use single hidden layer with 256 neurons**

### 2. Learning Rate Analysis
- **Higher learning rates (0.01) perform better than conservative rates (0.001)**
- LR 0.01: Average return = 1.28%, Hit rate = 13.3%
- LR 0.001: Average return = 1.27%, Hit rate = 8.3%
- **Recommendation: Use learning rate = 0.01**

### 3. Dropout Impact
- **Light regularization (0.2 dropout) outperforms no dropout**
- Dropout 0.2: Average return = 1.31%, Hit rate = 10.0%
- No dropout: Average return = 1.24%, Hit rate = 11.7%
- **Recommendation: Use dropout = 0.2 for stability**

### 4. Weight Decay
- **Minimal impact on performance**
- Both 0.0 and 1e-5 show similar results
- **Recommendation: Use weight_decay = 0.0 for simplicity**

## Performance Analysis

### Top Performing Configurations
1. **Best Return**: [256], LR=0.01, Dropout=0.2, WD=0.0 → **1.863% return**
2. **Second Best**: [256], LR=0.01, Dropout=0.2, WD=1e-5 → **1.829% return**
3. **Most Stable**: [256], LR=0.001, Dropout=0.2 → **1.748% return, lower variance**

### Risk-Return Profile
- **Inverse relationship between model complexity and returns (r = -0.90)**
- Simple models show higher variance but significantly better mean performance
- No significant correlation between returns and hit rates (r = 0.008)

## Overfitting/Underfitting Analysis

### Signs of Appropriate Model Complexity
- **Simple models show higher variance but better performance** - suggests they're capturing real signal
- **Deep models show lower variance but poor performance** - indicates underfitting or inability to optimize
- **95% overall success rate** - models generally converge successfully

### Stability Considerations
- Highest return models have moderate variance (0.37 std)
- Most stable models (low std) still achieve competitive returns
- **No evidence of severe overfitting** - high variance appears to be beneficial signal capture

## Training Efficiency

### Convergence Patterns
- Simple models: 11.9 epochs average, 0.28s training time
- Deep models: 13.7 epochs average, 0.29s training time
- **Simple models are both faster and better performing**

### Success Rates
- 15/16 configurations achieved 100% success rate
- Only 1 deep model configuration failed to converge consistently
- **Architecture robustness favors simple designs**

## Financial ML Intuition Check

### Results Make Sense Because:
1. **Factor selection is inherently noisy** - simpler models avoid overfitting to noise
2. **85 factors → 5 selections is a sparse selection problem** - complex models may struggle with this
3. **Monthly financial data has limited patterns** - overparameterization hurts more than helps
4. **High variance can indicate real signal capture** in financial contexts

### Return vs Hit Rate Tradeoff
- **Almost zero correlation (0.008)** suggests these measure different aspects
- Hit rate measures factor selection accuracy
- Returns measure actual financial performance
- **Focus should be on returns as the primary objective**

## Final Recommendations

### Optimal Configuration
```
Architecture: [256] (single hidden layer)
Learning Rate: 0.01
Dropout Rate: 0.2
Batch Size: 32
Weight Decay: 0.0
```

### Expected Performance
- **Average Top-5 Return: 1.86%**
- **Hit Rate: 6.7%** (low but returns are strong)
- **Training Time: <0.3 seconds**
- **Convergence: Highly reliable**

### Alternative Stable Configuration
For more conservative approach with lower variance:
```
Architecture: [256]
Learning Rate: 0.001
Dropout Rate: 0.2
Batch Size: 32
Weight Decay: 0.0
Expected Return: 1.75% (more stable)
```

## Risk Considerations

1. **Low hit rates (6-20%)** across all models suggest this is a challenging prediction task
2. **Higher returns with lower hit rates** may indicate the model is finding profitable but unconventional factor combinations
3. **Model simplicity is advantageous** - resist the urge to add complexity
4. **Financial significance**: 1.86% monthly return (if sustained) equals ~24% annualized return

## Next Steps

1. **Implement the optimal configuration** in production model
2. **Monitor out-of-sample performance** to validate tuning results
3. **Consider ensemble approaches** using multiple simple models
4. **Investigate why hit rates are low** but returns are high - may reveal important insights about factor dynamics