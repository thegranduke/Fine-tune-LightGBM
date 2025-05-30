import numpy as np
import lightgbm as lgb
import optuna
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from typing import Dict, List, Tuple
from sklearn.model_selection import LeaveOneGroupOut, GroupShuffleSplit
from sklearn.metrics import confusion_matrix

from data_loader import prepare_data
from feature_extractor import extract_features
from model import plot_confusion_matrix, plot_performance_metrics

def create_objective(X_train: np.ndarray, y_train: np.ndarray, subjects_train: np.ndarray):
    """
    Create an objective function for Optuna optimization.
    Uses LOSO cross-validation for robust evaluation.
    """
    def objective(trial: optuna.Trial) -> float:
        # Define the hyperparameter search space
        param = {
            'objective': 'multiclass',
            'metric': 'multi_logloss',
            'num_class': 6,
            'verbosity': -1,
            'boosting_type': 'gbdt',
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 16, 256),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'lambda_l1': trial.suggest_float('lambda_l1', 0.0, 5.0),
            'lambda_l2': trial.suggest_float('lambda_l2', 0.0, 5.0),
            'random_state': 42
        }
        
        # Initialize LOSO cross-validation
        logo = LeaveOneGroupOut()
        cv_scores = []
        
        # Use only a subset of folds to speed up optimization
        max_folds = 5  # Limit number of CV folds for faster optimization
        fold_count = 0
        
        for train_idx, val_idx in logo.split(X_train, y_train, subjects_train):
            if fold_count >= max_folds:
                break
                
            X_fold_train = X_train[train_idx]
            y_fold_train = y_train[train_idx]
            X_fold_val = X_train[val_idx]
            y_fold_val = y_train[val_idx]
            
            # Create LightGBM datasets
            train_data = lgb.Dataset(X_fold_train, y_fold_train)
            val_data = lgb.Dataset(X_fold_val, y_fold_val, reference=train_data)
            
            # Train with early stopping
            callbacks = [
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=0)  # Disable logging
            ]
            model = lgb.train(
                param,
                train_data,
                valid_sets=[val_data],
                num_boost_round=1000,
                callbacks=callbacks
            )
            
            # Predict and compute macro F1 score
            y_pred = np.argmax(model.predict(X_fold_val), axis=1)
            f1 = compute_macro_f1(y_fold_val, y_pred)
            cv_scores.append(f1)
            
            fold_count += 1
        
        # Return mean CV score
        return np.mean(cv_scores)
    
    return objective

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

def save_optimization_results(study: optuna.Study, output_dir: Path):
    """Save optimization results to files."""
    # Create detailed results text file
    with open(output_dir / 'results.txt', 'w') as f:
        # Study statistics
        f.write("Bayesian Optimization Results:\n")
        f.write("=" * 30 + "\n\n")
        
        f.write("Optimization Summary:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Number of trials: {len(study.trials)}\n")
        f.write(f"Best trial number: {study.best_trial.number}\n")
        f.write(f"Best score (macro F1): {study.best_value:.4f}\n\n")
        
        f.write("Best Hyperparameters:\n")
        f.write("-" * 20 + "\n")
        for param_name, param_value in study.best_params.items():
            f.write(f"{param_name}: {param_value}\n")
        f.write("\n")
        
        # Add hyperparameter importance if available
        try:
            importance = optuna.importance.get_param_importances(study)
            f.write("Hyperparameter Importance:\n")
            f.write("-" * 20 + "\n")
            for param_name, importance_score in importance.items():
                f.write(f"{param_name}: {importance_score:.4f}\n")
            f.write("\n")
        except:
            pass
        
        # Add trial history summary
        f.write("Top 10 Trials:\n")
        f.write("-" * 20 + "\n")
        sorted_trials = sorted(study.trials, key=lambda t: t.value if t.value is not None else float('-inf'), reverse=True)
        for i, trial in enumerate(sorted_trials[:10]):
            if trial.value is not None:
                f.write(f"Trial {trial.number}: Score = {trial.value:.4f}\n")
                for param_name, param_value in trial.params.items():
                    f.write(f"    {param_name}: {param_value}\n")
                f.write("\n")
    
    # Save best parameters as JSON
    with open(output_dir / 'best_params.json', 'w') as f:
        json.dump(study.best_params, f, indent=4)
    
    # Save all trials data
    trials_df = study.trials_dataframe()
    trials_df.to_csv(output_dir / 'optimization_history.csv', index=False)
    
    # Create visualization plots
    plt.figure(figsize=(12, 6))
    optuna.visualization.matplotlib.plot_optimization_history(study)
    plt.title('Optimization History')
    plt.tight_layout()
    plt.savefig(output_dir / 'optimization_history.png')
    plt.close()
    
    plt.figure(figsize=(12, 6))
    optuna.visualization.matplotlib.plot_param_importances(study)
    plt.title('Hyperparameter Importance')
    plt.tight_layout()
    plt.savefig(output_dir / 'param_importances.png')
    plt.close()
    
    try:
        plt.figure(figsize=(12, 8))
        optuna.visualization.matplotlib.plot_parallel_coordinate(study)
        plt.title('Parallel Coordinate Plot')
        plt.tight_layout()
        plt.savefig(output_dir / 'parallel_coordinate.png')
        plt.close()
    except:
        pass

def evaluate_best_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    subjects_train: np.ndarray,
    subjects_test: np.ndarray,
    best_params: Dict,
    output_dir: Path
):
    """Evaluate the best model and save results."""
    # Update parameters for final training
    final_params = {
        'objective': 'multiclass',
        'metric': 'multi_logloss',
        'num_class': 6,
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'random_state': 42,
        **best_params
    }
    
    # Train final model
    train_data = lgb.Dataset(X_train, y_train)
    final_model = lgb.train(final_params, train_data)
    
    # Evaluate on test set
    y_pred = np.argmax(final_model.predict(X_test), axis=1)
    test_accuracy = np.mean(y_test == y_pred)
    test_f1 = compute_macro_f1(y_test, y_pred)
    
    # Get detailed classification report
    from sklearn.metrics import classification_report
    class_names = ['Downstairs', 'Jogging', 'Sitting', 'Standing', 'Upstairs', 'Walking']
    classification_rep = classification_report(y_test, y_pred, target_names=class_names)
    
    # Compute confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Perform LOSO cross-validation for final evaluation
    logo = LeaveOneGroupOut()
    cv_accuracies = []
    cv_f1_scores = []
    cv_matrices = []
    
    for train_idx, val_idx in logo.split(X_train, y_train, subjects_train):
        X_fold_train = X_train[train_idx]
        y_fold_train = y_train[train_idx]
        X_fold_val = X_train[val_idx]
        y_fold_val = y_train[val_idx]
        
        # Train model
        train_data = lgb.Dataset(X_fold_train, y_fold_train)
        model = lgb.train(final_params, train_data)
        
        # Evaluate
        y_fold_pred = np.argmax(model.predict(X_fold_val), axis=1)
        cv_accuracies.append(np.mean(y_fold_val == y_fold_pred))
        cv_f1_scores.append(compute_macro_f1(y_fold_val, y_fold_pred))
        cv_matrices.append(confusion_matrix(y_fold_val, y_fold_pred))
    
    # Save detailed results
    with open(output_dir / 'model_evaluation.txt', 'w') as f:
        f.write("Model Evaluation Results:\n")
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
    """Main function to run Bayesian optimization."""
    # Create output directory
    output_dir = Path('results/bayesian_opt')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load and prepare data
    print("Loading and preparing data...")
    X_raw, y, subjects = prepare_data()
    
    # Extract features
    print("Extracting features...")
    X = extract_features(X_raw)
    
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
    
    # Create study
    print("Starting Bayesian optimization...")
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner()
    )
    
    # Run optimization
    study.optimize(
        create_objective(X_train, y_train, subjects_train),
        n_trials=50,  # Reduced number of trials for faster optimization
        n_jobs=-1,
        show_progress_bar=True
    )
    
    print("\nBest trial:")
    print(f"  Value: {study.best_value:.4f}")
    print("  Params:")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")
    
    # Save optimization results
    print("\nSaving optimization results...")
    save_optimization_results(study, output_dir)
    
    # Evaluate best model
    print("\nEvaluating best model...")
    evaluate_best_model(
        X_train, y_train, X_test, y_test,
        subjects_train, subjects_test,
        study.best_params, output_dir
    )
    
    print(f"\nDone! Results saved to: {output_dir}")

if __name__ == '__main__':
    main() 