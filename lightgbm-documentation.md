# LightGBM Classifier for Human Activity Recognition (MotionSense)

## Overview
This documentation describes the configuration, execution steps, and methodology used in training a LightGBM classifier on the MotionSense dataset for human activity recognition.

## How to Run

### Step 1: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run model training and evaluation
```bash
python src/main.py --results_subdir lightgbm_base --test_size 0.2 --data_dir motion-sense/data
```

## Model Configuration
- **Model Type**: LightGBM (LGBMClassifier)
- **Boosting Type**: gbdt
- **Objective**: multiclass
- **Num Classes**: 6
- **Learning Rate**: 0.1
- **n_estimators**: 100
- **Random State**: 42

## Data Handling
- **Dataset**: MotionSense accelerometer and gyroscope data
- **Window Size**: 100 samples
- **Overlap**: 50%
- **Normalization**: Per-subject, per-axis

## Feature Engineering
### Statistical Features
- Mean
- Standard Deviation
- Min
- Max
- Root Mean Square (RMS)
- Signal Magnitude Area (SMA)

### Spectral Features
- Spectral Energy
- Dominant Frequency
- Mean Frequency
- Frequency Variance

## Evaluation
- **Train/Test Split**: GroupShuffleSplit by subject (80%/20%)
- **Cross-Validation**: Leave-One-Subject-Out (LOSO)
- **Metrics**:
  - Accuracy
  - Macro F1-score
  - Confusion Matrix
  - Per-class precision/recall/F1

## Results Summary (Updated After Each Run)
| Metric | Value |
|--------|-------|
| Test Accuracy | TBD |
| Test Macro F1-score | TBD |
| Mean CV Accuracy | TBD |
| Mean CV Macro F1 | TBD |

## Update Log
- v1: Initial file structure and training logic created.
- Future runs should append config changes, tuning decisions, and result variations here.

# LightGBM Model Documentation

## Feature Pruning

The feature pruning process helps reduce model complexity and potentially improve generalization by removing less important features based on their contribution to the model's predictions.

### Methodology

1. **Initial Model Training**
   - Train a baseline LightGBM model using all features
   - Use Leave-One-Subject-Out (LOSO) cross-validation to ensure robust feature importance estimates
   - Evaluate performance metrics (accuracy, F1-score) for baseline comparison

2. **Feature Importance Computation**
   - Use LightGBM's built-in feature importance scores
   - Normalize importance scores to sum to 1.0
   - Generate feature importance plots for visualization

3. **Feature Selection**
   - Rank features by importance scores
   - Remove bottom X% of features (configurable via `--prune_bottom_percent`)
   - Save lists of retained and dropped features for analysis

4. **Model Retraining**
   - Train new model using only retained features
   - Evaluate using:
     - GroupShuffleSplit (80/20 subject-based test split)
     - Leave-One-Subject-Out cross-validation
   - Compare performance with baseline model

### Usage

Run the feature pruning script with:

```bash
python src/prune_features_lightgbm.py
```

Configuration parameters are defined at the top of the `main()` function in `src/prune_features_lightgbm.py`:

```python
# Path to the MotionSense dataset directory
DATA_DIR = 'motion-sense/data'

# Window size for segmentation (number of samples)
WINDOW_SIZE = 100

# Overlap between consecutive windows (0-1)
OVERLAP = 0.5

# Percentage of least important features to remove (0-1)
# Example: 0.2 means remove bottom 20% of features
PRUNE_BOTTOM_PERCENT = 0.2

# Directory to save all results and plots
OUTPUT_DIR = 'results/feature_pruning'
```

To modify any of these parameters, simply edit their values in the code.

### Output Files

The script generates the following outputs in the specified output directory:

1. **Feature Lists**
   - `important_features.txt`: List of retained features
   - `dropped_features.txt`: List of removed features

2. **Visualizations**
   - `feature_importance_plot.png`: Bar plot of top feature importances
   - `confusion_matrix(test_set).png`: Confusion matrix on test set
   - `confusion_matrix(cross-validation).png`: Average confusion matrix from CV
   - `performance_metrics(cross-validation).png`: Box plots of CV metrics

3. **Performance Metrics**
   - `results.txt`: JSON file containing:
     - Test set accuracy and macro F1-score
     - Cross-validation metrics (mean/std of accuracy and F1-score)

### Interpreting Results

1. **Feature Importance Plot**
   - Examine which features contribute most to predictions
   - Look for patterns in types of important features (e.g., time vs frequency domain)

2. **Performance Comparison**
   - Compare pruned model metrics with baseline
   - Check if removing features affects specific activity recognition

3. **Cross-validation Results**
   - Use CV metrics to assess generalization across subjects
   - Higher mean with lower std suggests robust performance

### Best Practices

1. **Feature Selection**
   - Start with conservative pruning (e.g., 20%)
   - Gradually increase if performance maintains/improves
   - Consider domain knowledge when interpreting important features

2. **Validation**
   - Always use subject-based splitting to prevent data leakage
   - Compare multiple pruning thresholds if time permits
   - Consider impact on inference speed vs accuracy trade-off

3. **Monitoring**
   - Track changes in feature importance rankings across runs
   - Watch for stability in important feature sets
   - Monitor impact on specific activity recognition performance

## Bayesian Optimization

This section describes the hyperparameter optimization process using Bayesian Optimization with Optuna.

### Methodology

1. **Optimization Setup**
   - Framework: Optuna with TPE (Tree-structured Parzen Estimators) sampler
   - Objective: Maximize macro F1-score
   - Cross-validation: Leave-One-Subject-Out (LOSO)
   - Early stopping: 50 rounds
   - Number of trials: 50
   - Parallel execution: n_jobs=-1

2. **Hyperparameter Search Space**
   ```python
   {
       'learning_rate': [0.005, 0.3],        # log-uniform
       'num_leaves': [16, 256],              # int
       'max_depth': [3, 20],                 # int
       'min_child_samples': [5, 50],         # int
       'subsample': [0.5, 1.0],              # float
       'colsample_bytree': [0.5, 1.0],       # float
       'lambda_l1': [0.0, 5.0],              # float
       'lambda_l2': [0.0, 5.0],              # float
   }
   ```

3. **Optimization Process**
   - Uses TPE algorithm for efficient exploration
   - Implements pruning for early termination of poor trials
   - Limited to 5 LOSO folds during optimization for speed
   - Full LOSO evaluation on best parameters

4. **Model Evaluation**
   - Final evaluation uses:
     - GroupShuffleSplit (80/20) for test set
     - Complete LOSO cross-validation
   - Metrics computed:
     - Accuracy
     - Macro F1-score
     - Per-class metrics
     - Confusion matrices

### Usage

Run the optimization script with:

```bash
python src/tune_lightgbm_bayesian.py
```

### Output Files

The script generates the following in `results/bayesian_opt/`:

1. **Optimization Results**
   - `best_params.json`: Best hyperparameters found
   - `bayes_opt_trials.csv`: All trial results
   - `optimization_history.png`: Convergence plot

2. **Model Evaluation**
   - `results.txt`: Final performance metrics
   - `confusion_matrix(test_set).png`
   - `confusion_matrix(cross-validation).png`
   - `performance_metrics(cross-validation).png`

### Performance Optimizations

1. **Speed Improvements**
   - Limited LOSO folds during optimization
   - Early stopping for each trial
   - Parallel trial execution
   - Pruning of poor-performing trials

2. **Robustness**
   - Cross-validation ensures generalization
   - Multiple metrics for comprehensive evaluation
   - Separate test set for unbiased assessment

### Best Practices

1. **Optimization**
   - Start with wide parameter ranges
   - Use log-uniform for learning rate
   - Monitor convergence via history plot
   - Consider trade-off between trials and CV folds

2. **Evaluation**
   - Compare with baseline model
   - Check for overfitting
   - Analyze per-class performance
   - Consider model complexity vs. performance

3. **Production Use**
   - Save best parameters for reuse
   - Document performance characteristics
   - Monitor training time and resource usage
   - Consider model size and inference speed 