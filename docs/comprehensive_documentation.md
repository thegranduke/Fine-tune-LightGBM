# LightGBM-based Human Activity Recognition System Documentation

## Table of Contents
1. [Overview](#overview)
2. [Data Processing Pipeline](#data-processing-pipeline)
3. [Base Model Implementation](#base-model-implementation)
4. [Feature Pruning Approach](#feature-pruning-approach)
5. [Bayesian Optimization](#bayesian-optimization)
6. [Performance Analysis](#performance-analysis)

## Overview
This project implements a human activity recognition system using the MotionSense dataset and LightGBM classifier. The system is capable of classifying six different activities:
- Walking
- Jogging
- Sitting
- Standing
- Walking Upstairs
- Walking Downstairs

The implementation includes three main approaches:
1. Base LightGBM model with comprehensive feature engineering
2. Feature importance-based pruning for model optimization
3. Bayesian optimization for hyperparameter tuning

## Data Processing Pipeline

### 1. Data Loading and Preprocessing

#### Raw Data Structure
- **Source**: MotionSense dataset
- **Sensors**:
  - Accelerometer (3 channels: x, y, z)
  - Gyroscope (3 channels: x, y, z)
- **Sampling Rate**: ~50Hz
- **Data Format**: Time series data for each sensor channel

#### Preprocessing Steps

1. **Initial Data Loading** (`data_loader.py`):
   - Synchronous loading of accelerometer and gyroscope data
   - Timestamp matching between sensor types
   - Combination into 6-dimensional time series
   - Activity label mapping: {dws: 0, jog: 1, sit: 2, std: 3, ups: 4, wlk: 5}

2. **Data Segmentation**:
   - Window Size: 100 samples (~2 seconds of data)
   - Overlap: 50% between consecutive windows
   - Purpose: Capture temporal patterns while ensuring sufficient training samples
   - Implementation: Sliding window approach with majority voting for labels

3. **Per-subject Normalization**:
   - Method: Standard scaling (zero mean, unit variance)
   - Applied separately for each subject
   - Purpose: Account for individual differences in sensor readings
   - Implementation: Using sklearn's StandardScaler

### 2. Feature Engineering

#### Time Domain Features
For each sensor channel:
- Mean value
- Standard deviation
- Minimum and maximum values
- Root Mean Square (RMS)
- Signal Magnitude Area (SMA)
  - Separate calculation for accelerometer and gyroscope

#### Frequency Domain Features
For each sensor channel:
- Spectral energy
- Dominant frequency
- Mean frequency
- Frequency variance
- FFT-based features

## Base Model Implementation

### LightGBM Configuration
- Model Type: LightGBM Classifier
- Objective: Multiclass classification
- Number of Classes: 6
- Boosting Type: GBDT (Gradient Boosting Decision Tree)
- Base Parameters:
  - Learning Rate: 0.1
  - Number of Estimators: 100
  - Random State: 42

### Training Approach
1. **Data Split**:
   - Method: GroupShuffleSplit
   - Test Size: 20%
   - Grouping: By subject ID (ensures subject independence)

2. **Cross-validation**:
   - Method: Leave-One-Subject-Out (LOSO)
   - Purpose: Robust evaluation of model generalization
   - Implementation: Custom LOSO cross-validation loop

## Feature Pruning Approach

### Methodology
1. **Feature Importance Calculation**:
   - Using LightGBM's built-in feature importance
   - Importance type: 'gain'
   - Averaged across LOSO cross-validation folds

2. **Pruning Process**:
   - Bottom-up approach
   - Removes least important features (bottom 20%)
   - Retains features with consistent importance across folds

### Important Features (Top 10)
1. mean_1 (Channel 1 mean)
2. mean_4 (Channel 4 mean)
3. max_1 (Channel 1 maximum)
4. mean_2 (Channel 2 mean)
5. std_1 (Channel 1 standard deviation)
6. mean_freq_2 (Channel 2 mean frequency)
7. min_1 (Channel 1 minimum)
8. min_2 (Channel 2 minimum)
9. max_5 (Channel 5 maximum)
10. mean_0 (Channel 0 mean)

### Dropped Features
Primarily frequency-domain features with low importance:
- Frequency variance features
- Spectral energy features
- Some RMS features

## Bayesian Optimization

### Optimization Framework
- **Tool**: Optuna
- **Sampler**: TPESampler (Tree-structured Parzen Estimators)
- **Pruner**: MedianPruner
- **Number of Trials**: 50

### Hyperparameter Search Space
- learning_rate: [0.005, 0.3] (log scale)
- num_leaves: [16, 256]
- max_depth: [3, 20]
- min_child_samples: [5, 50]
- subsample: [0.5, 1.0]
- colsample_bytree: [0.5, 1.0]
- lambda_l1: [0.0, 5.0]
- lambda_l2: [0.0, 5.0]

### Optimization Process
1. **Objective Function**:
   - Uses LOSO cross-validation
   - Limited to 5 folds for speed
   - Optimizes macro F1 score
   - Early stopping after 50 rounds without improvement

2. **Evaluation Metrics**:
   - Primary: Macro F1 score
   - Secondary: Accuracy
   - Cross-validation performance
   - Per-class metrics

## Performance Analysis

### Evaluation Metrics
1. **Classification Accuracy**:
   - Overall accuracy
   - Per-class accuracy
   - Confusion matrix

2. **F1 Scores**:
   - Macro F1 score
   - Per-class F1 scores

3. **Cross-validation Performance**:
   - Mean and standard deviation of metrics
   - Per-fold results
   - Subject-wise analysis

### Model Comparison
1. **Base Model**:
   - Full feature set
   - Default hyperparameters
   - Baseline performance

2. **Pruned Model**:
   - Reduced feature set (80% of original features)
   - Improved efficiency
   - Comparable performance to base model

3. **Optimized Model**:
   - Tuned hyperparameters
   - Best performance
   - Optimal balance of accuracy and efficiency

### Usage Guidelines
1. **Base Model**: Use when:
   - Quick implementation needed
   - All features potentially important
   - Computational resources not constrained

2. **Pruned Model**: Use when:
   - Faster inference needed
   - Resource constraints present
   - Feature interpretability important

3. **Optimized Model**: Use when:
   - Maximum performance required
   - Training time not constrained
   - Fine-tuning needed for specific use case 