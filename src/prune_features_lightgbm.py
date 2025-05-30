import numpy as np
import lightgbm as lgb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Set
import json
from sklearn.model_selection import LeaveOneGroupOut, GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix

from data_loader import prepare_data
from feature_extractor import extract_features, get_feature_names
from model import train_and_evaluate_lightgbm, plot_confusion_matrix, plot_performance_metrics

def compute_macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute macro F1 score."""
    # Get unique classes
    classes = np.unique(y_true)
    f1_scores = []
    
    for cls in classes:
        true_pos = np.sum((y_true == cls) & (y_pred == cls))
        false_pos = np.sum((y_true != cls) & (y_pred == cls))
        false_neg = np.sum((y_true == cls) & (y_pred != cls))
        
        precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
        recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
        
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        f1_scores.append(f1)
    
    return np.mean(f1_scores)

def get_feature_importance(
    X_train: np.ndarray,
    y_train: np.ndarray,
    subjects_train: np.ndarray,
    feature_names: List[str]
) -> pd.DataFrame:
    """Get feature importance scores using LOSO cross-validation."""
    # Model parameters
    params = {
        'objective': 'multiclass',
        'metric': 'multi_logloss',
        'num_class': 6,
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'learning_rate': 0.1,
        'num_leaves': 31,
        'random_state': 42
    }
    
    # Initialize LOSO cross-validation
    logo = LeaveOneGroupOut()
    importance_scores = []
    
    for train_idx, val_idx in logo.split(X_train, y_train, subjects_train):
        X_fold_train = X_train[train_idx]
        y_fold_train = y_train[train_idx]
        
        # Train model
        train_data = lgb.Dataset(X_fold_train, y_fold_train)
        model = lgb.train(params, train_data)
        
        # Get feature importance
        importance = model.feature_importance(importance_type='gain')
        importance_scores.append(importance)
    
    # Average importance scores across folds
    mean_importance = np.mean(importance_scores, axis=0)
    std_importance = np.std(importance_scores, axis=0)
    
    # Create DataFrame with feature importance stats
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance_mean': mean_importance,
        'importance_std': std_importance
    })
    
    # Normalize importance scores
    importance_df['importance_mean'] = importance_df['importance_mean'] / importance_df['importance_mean'].sum()
    
    return importance_df.sort_values('importance_mean', ascending=False)

def save_pruning_results(
    importance_df: pd.DataFrame,
    retained_features: Set[str],
    dropped_features: Set[str],
    output_dir: Path,
    prune_threshold: float
):
    """Save feature pruning results and visualizations."""
    # Save detailed results
    with open(output_dir / 'feature_pruning_results.txt', 'w') as f:
        f.write("Feature Pruning Results:\n")
        f.write("=" * 30 + "\n\n")
        
        f.write("Pruning Summary:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Total features: {len(importance_df)}\n")
        f.write(f"Retained features: {len(retained_features)}\n")
        f.write(f"Dropped features: {len(dropped_features)}\n")
        f.write(f"Pruning threshold: {prune_threshold:.2%}\n\n")
        
        f.write("Top 20 Most Important Features:\n")
        f.write("-" * 20 + "\n")
        top_20 = importance_df.head(20)
        for _, row in top_20.iterrows():
            f.write(f"{row['feature']}: {row['importance_mean']:.4f} (±{row['importance_std']:.4f})\n")
        f.write("\n")
        
        f.write("Dropped Features:\n")
        f.write("-" * 20 + "\n")
        dropped_df = importance_df[importance_df['feature'].isin(dropped_features)]
        for _, row in dropped_df.iterrows():
            f.write(f"{row['feature']}: {row['importance_mean']:.4f} (±{row['importance_std']:.4f})\n")
    
    # Save feature lists
    with open(output_dir / 'retained_features.txt', 'w') as f:
        for feature in sorted(retained_features):
            f.write(f"{feature}\n")
    
    with open(output_dir / 'dropped_features.txt', 'w') as f:
        for feature in sorted(dropped_features):
            f.write(f"{feature}\n")
    
    # Plot feature importance
    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=importance_df.head(20),
        x='importance_mean',
        y='feature',
        palette='viridis'
    )
    plt.title('Top 20 Feature Importance Scores')
    plt.xlabel('Normalized Importance')
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_importance.png')
    plt.close()

def evaluate_pruned_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    subjects_train: np.ndarray,
    subjects_test: np.ndarray,
    retained_feature_indices: np.ndarray,
    output_dir: Path
):
    """Evaluate model performance after feature pruning."""
    # Select retained features
    X_train_pruned = X_train[:, retained_feature_indices]
    X_test_pruned = X_test[:, retained_feature_indices]
    
    # Model parameters
    params = {
        'objective': 'multiclass',
        'metric': 'multi_logloss',
        'num_class': 6,
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'learning_rate': 0.1,
        'num_leaves': 31,
        'random_state': 42
    }
    
    # Train final model
    train_data = lgb.Dataset(X_train_pruned, y_train)
    model = lgb.train(params, train_data)
    
    # Evaluate on test set
    y_pred = np.argmax(model.predict(X_test_pruned), axis=1)
    test_accuracy = np.mean(y_test == y_pred)
    test_f1 = compute_macro_f1(y_test, y_pred)
    
    # Get detailed classification report
    class_names = ['Downstairs', 'Jogging', 'Sitting', 'Standing', 'Upstairs', 'Walking']
    classification_rep = classification_report(y_test, y_pred, target_names=class_names)
    
    # Compute confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Perform LOSO cross-validation
    logo = LeaveOneGroupOut()
    cv_accuracies = []
    cv_f1_scores = []
    cv_matrices = []
    
    for train_idx, val_idx in logo.split(X_train_pruned, y_train, subjects_train):
        X_fold_train = X_train_pruned[train_idx]
        y_fold_train = y_train[train_idx]
        X_fold_val = X_train_pruned[val_idx]
        y_fold_val = y_train[val_idx]
        
        # Train model
        train_data = lgb.Dataset(X_fold_train, y_fold_train)
        model = lgb.train(params, train_data)
        
        # Evaluate
        y_fold_pred = np.argmax(model.predict(X_fold_val), axis=1)
        cv_accuracies.append(np.mean(y_fold_val == y_fold_pred))
        cv_f1_scores.append(compute_macro_f1(y_fold_val, y_fold_pred))
        cv_matrices.append(confusion_matrix(y_fold_val, y_fold_pred))
    
    # Save detailed results
    with open(output_dir / 'model_evaluation.txt', 'w') as f:
        f.write("Pruned Model Evaluation Results:\n")
        f.write("=" * 30 + "\n\n")
        
        f.write("Test Set Results:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Accuracy: {test_accuracy:.4f}\n")
        f.write(f"Macro F1: {test_f1:.4f}\n\n")
        
        f.write("Classification Report:\n")
        f.write("-" * 20 + "\n")
        f.write(classification_rep)
        f.write("\n")
        
        f.write("Cross-validation Results:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Mean Accuracy: {np.mean(cv_accuracies):.4f} (±{np.std(cv_accuracies):.4f})\n")
        f.write(f"Mean Macro F1: {np.mean(cv_f1_scores):.4f} (±{np.std(cv_f1_scores):.4f})\n")
        f.write(f"Number of CV folds: {len(cv_accuracies)}\n\n")
        
        f.write("Per-fold Results:\n")
        f.write("-" * 20 + "\n")
        for i, (acc, f1) in enumerate(zip(cv_accuracies, cv_f1_scores)):
            f.write(f"Fold {i+1}:\n")
            f.write(f"  Accuracy: {acc:.4f}\n")
            f.write(f"  Macro F1: {f1:.4f}\n")
    
    # Plot confusion matrices
    plot_confusion_matrix(
        conf_matrix,
        'Confusion Matrix (Test Set)',
        class_names=class_names,
        save_path=output_dir / 'confusion_matrix_test.png'
    )
    
    avg_cv_matrix = np.mean(cv_matrices, axis=0)
    plot_confusion_matrix(
        avg_cv_matrix,
        'Average Confusion Matrix (Cross-validation)',
        class_names=class_names,
        save_path=output_dir / 'confusion_matrix_cv.png'
    )
    
    # Plot CV performance metrics
    cv_scores = {
        'accuracies': cv_accuracies,
        'f1_scores': cv_f1_scores,
        'confusion_matrices': cv_matrices
    }
    plot_performance_metrics(
        cv_scores,
        save_path=output_dir / 'cv_performance_metrics.png'
    )

def main():
    """
    Main function to run feature pruning.
    
    To modify the configuration, adjust the parameters below:
    """
    # ============ CONFIGURATION PARAMETERS ============
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
    # ===============================================
    
    # Create output directory
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load and prepare data
    print("Loading and preparing data...")
    X_raw, y, subjects = prepare_data(DATA_DIR, WINDOW_SIZE, OVERLAP)
    
    # Extract features
    print("Extracting features...")
    X, feature_names = extract_features(X_raw, return_feature_names=True)
    
    # Split data
    print("Splitting data...")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, subjects))
    
    X_train = X[train_idx]
    y_train = y[train_idx]
    subjects_train = subjects[train_idx]
    X_test = X[test_idx]
    y_test = y[test_idx]
    subjects_test = subjects[test_idx]
    
    # Get feature importance
    print("Computing feature importance...")
    importance_df = get_feature_importance(X_train, y_train, subjects_train, feature_names)
    
    # Determine features to retain/drop
    n_features = len(feature_names)
    n_retain = int(n_features * (1 - PRUNE_BOTTOM_PERCENT))
    retained_features = set(importance_df['feature'].head(n_retain))
    dropped_features = set(importance_df['feature'].tail(n_features - n_retain))
    
    # Save pruning results
    print("Saving pruning results...")
    save_pruning_results(
        importance_df,
        retained_features,
        dropped_features,
        output_dir,
        PRUNE_BOTTOM_PERCENT
    )
    
    # Get indices of retained features
    retained_indices = np.array([
        i for i, name in enumerate(feature_names)
        if name in retained_features
    ])
    
    # Evaluate pruned model
    print("Evaluating pruned model...")
    evaluate_pruned_model(
        X_train, y_train,
        X_test, y_test,
        subjects_train, subjects_test,
        retained_indices,
        output_dir
    )
    
    print(f"\nDone! Results saved to: {output_dir}")

if __name__ == '__main__':
    main() 