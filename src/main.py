import os
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit

from data_loader import prepare_data
from feature_extractor import extract_features
from model import train_and_evaluate_lightgbm, plot_confusion_matrix, plot_performance_metrics, ACTIVITY_LABELS

def main():
    # Configuration parameters
    # These could be moved to a config.py file in the future if needed
    RESULTS_SUBDIR = 'lightgbm_base'  # Subdirectory within results/ to save outputs
    TEST_SIZE = 0.2  # Proportion of subjects to use for testing
    
    # Set up paths relative to the project root
    project_root = Path(__file__).parent.parent
    data_dir = project_root / 'motion-sense' / 'data'
    results_dir = project_root / 'results' / RESULTS_SUBDIR
    
    # Create results directory
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    X, y, subject_ids = prepare_data(data_dir=str(data_dir))
    
    # Extract features
    print("Extracting features...")
    X_features = extract_features(X)
    
    # Split data by subject
    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=42)
    train_idx, test_idx = next(splitter.split(X_features, y, subject_ids))
    
    X_train = X_features[train_idx]
    y_train = y[train_idx]
    subject_train = subject_ids[train_idx]
    
    X_test = X_features[test_idx]
    y_test = y[test_idx]
    subject_test = subject_ids[test_idx]
    
    # Train and evaluate model
    print("Training and evaluating model...")
    model, metrics = train_and_evaluate_lightgbm(
        X_train, y_train,
        X_test, y_test,
        subject_train, subject_test
    )
    
    # Save results
    print("Saving results...")
    
    # Save confusion matrices
    plot_confusion_matrix(
        metrics['confusion_matrix'],
        'Test Set Confusion Matrix',
        save_path=results_dir / 'confusion_matrix(test_set).png'
    )
    
    if 'cv_scores' in metrics:
        # Average CV confusion matrices
        cv_cm_mean = np.mean(metrics['cv_scores']['confusion_matrices'], axis=0)
        plot_confusion_matrix(
            cv_cm_mean,
            'Cross-validation Confusion Matrix (Mean)',
            save_path=results_dir / 'confusion_matrix(cross-validation).png'
        )
        
        # Plot CV performance metrics
        plot_performance_metrics(
            metrics['cv_scores'],
            save_path=results_dir / 'performance_metrics(cross-validation).png'
        )
    
    # Save text results
    with open(results_dir / 'results.txt', 'w') as f:
        f.write("Test Set Results:\n")
        f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"Macro F1: {metrics['macro_f1']:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(metrics['classification_report'])
        f.write("\n")
        
        if 'cv_scores' in metrics:
            f.write("\nCross-validation Results:\n")
            f.write(f"Mean Accuracy: {metrics['cv_scores']['mean_accuracy']:.4f} "
                   f"(±{metrics['cv_scores']['std_accuracy']:.4f})\n")
            f.write(f"Mean Macro F1: {metrics['cv_scores']['mean_f1']:.4f} "
                   f"(±{metrics['cv_scores']['std_f1']:.4f})\n")
    
    print("Done! Results saved in:", results_dir)

if __name__ == '__main__':
    main() 