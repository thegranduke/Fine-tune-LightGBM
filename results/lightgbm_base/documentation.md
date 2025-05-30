# LightGBM-based Human Activity Recognition: Implementation Documentation

## Overview
This document details the implementation of a LightGBM-based classifier for human activity recognition using the MotionSense dataset. The system processes raw accelerometer and gyroscope data to classify six different activities: walking, jogging, sitting, standing, walking upstairs, and walking downstairs.

## Data Processing Pipeline

### 1. Data Loading and Preprocessing
#### Raw Data Structure
- Source: MotionSense dataset containing accelerometer and gyroscope readings
- Data organization:
  - Accelerometer data: 3 channels (x, y, z)
  - Gyroscope data: 3 channels (x, y, z)
  - Sampling rate: ~50Hz
  - Activities: 6 classes (walking, jogging, sitting, standing, upstairs, downstairs)

#### Preprocessing Steps
1. **Data Loading**
   - Synchronous loading of accelerometer and gyroscope data
   - Matching of timestamps between both sensor types
   - Combination of sensor data into a 6-dimensional time series (3 acc + 3 gyro)

2. **Windowing**
   - Window size: 100 samples (~2 seconds of data)
   - Overlap: 50% between consecutive windows
   - Purpose: Capture temporal patterns while maintaining sufficient training samples

3. **Per-subject Normalization**
   - Standard scaling (zero mean, unit variance) applied separately for each subject
   - Rationale: Accounts for individual differences in sensor readings and movement patterns
   - Implementation: Using sklearn's StandardScaler

### 2. Feature Extraction
A comprehensive set of statistical and spectral features are extracted from each window:

#### Time Domain Features
- Mean, standard deviation, median
- Maximum and minimum values
- Root mean square (RMS)
- Zero crossing rate
- Mean absolute deviation
- Interquartile range
- Skewness and kurtosis
- Signal magnitude area

#### Frequency Domain Features
- FFT coefficients (first 10 components)
- Spectral entropy
- Dominant frequency
- Frequency band energy
- Spectral centroid

Features are extracted independently for each sensor axis, resulting in a rich representation of the movement patterns.

## Model Implementation

### 1. LightGBM Configuration
- Model type: Gradient Boosting Decision Tree
- Objective: Multi-class classification
- Number of classes: 6
- Number of estimators: 100
- Learning rate: 0.1
- Boosting type: gbdt (Gradient Boosting Decision Tree)

### 2. Cross-validation Strategy
- Method: Leave-One-Subject-Out (LOSO) cross-validation
- Rationale: Tests model's generalization to unseen subjects
- Implementation: 
  - Each fold leaves out one subject for validation
  - Model is trained on remaining subjects
  - Performance is evaluated on the held-out subject

### 3. Train-Test Split
- Split ratio: 80% training, 20% testing
- Split method: Group-based (by subject) using GroupShuffleSplit
- Purpose: Ensures no data leakage between subjects

## Results Analysis

### Test Set Performance
- Overall Accuracy: 90.74%
- Macro F1-score: 87.89%

#### Per-Activity Performance
1. Static Activities
   - Sitting: Perfect classification (F1: 1.00)
   - Standing: Perfect classification (F1: 1.00)
   - Analysis: Model excels at distinguishing static postures

2. Dynamic Activities
   - Jogging: Excellent performance (F1: 0.98)
   - Walking: Very good performance (F1: 0.87)
   - Analysis: Clear distinction between different locomotion speeds

3. Stair Activities
   - Downstairs: Moderate performance (F1: 0.64)
   - Upstairs: Good performance (F1: 0.78)
   - Analysis: Some confusion between stair-related activities, possibly due to similar motion patterns

### Cross-validation Results
- Mean Accuracy: 92.92% (±6.46%)
- Mean Macro F1: 91.95% (±6.17%)
- Analysis: Strong performance across different subjects with reasonable variance

## Discussion and Insights

### Strengths
1. **Robust Feature Engineering**
   - Comprehensive feature set captures both temporal and spectral characteristics
   - Per-subject normalization helps handle individual variations

2. **Model Performance**
   - Excellent performance on static activities
   - Strong discrimination between different locomotion types
   - Good generalization across subjects (shown by cross-validation)

### Limitations and Potential Improvements
1. **Stair Activity Classification**
   - Lower performance on stair-related activities
   - Potential solutions:
     - Additional features specific to vertical movement
     - Increased window size for better pattern capture
     - Hierarchical classification approach

2. **Subject Variability**
   - Cross-validation standard deviation (±6.46%) indicates some subject-dependent variation
   - Could be addressed by:
     - More sophisticated normalization techniques
     - Subject-adaptive feature extraction
     - Ensemble methods with subject-specific models

### Future Directions
1. **Feature Selection/Optimization**
   - Analyze feature importance
   - Remove redundant features
   - Add domain-specific features for challenging activities

2. **Model Enhancements**
   - Hyperparameter optimization
   - Deep learning integration for automatic feature learning
   - Real-time prediction optimization

3. **Data Augmentation**
   - Synthetic data generation
   - Additional sensor modalities
   - More diverse subject population

## Technical Implementation Details
The implementation is organized into several Python modules:
- `data_loader.py`: Data loading and preprocessing
- `feature_extractor.py`: Feature extraction pipeline
- `model.py`: LightGBM model implementation and evaluation
- `main.py`: Orchestration and experiment running

All code is version controlled and documented for reproducibility. Results, including confusion matrices and performance metrics, are automatically saved in the results directory for analysis and comparison. 