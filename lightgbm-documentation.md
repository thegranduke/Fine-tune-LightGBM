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