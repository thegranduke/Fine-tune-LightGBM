import numpy as np
import lightgbm as lgb
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import LeaveOneGroupOut, GroupShuffleSplit
from typing import Dict, Tuple, Any, List
import matplotlib.pyplot as plt
import seaborn as sns

# Add activity labels
ACTIVITY_LABELS = ['Downstairs', 'Jogging', 'Sitting', 'Standing', 'Upstairs', 'Walking']

def train_and_evaluate_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    subject_train: np.ndarray = None,
    subject_test: np.ndarray = None
) -> Tuple[lgb.LGBMClassifier, Dict[str, Any]]:
    """
    Train and evaluate a LightGBM classifier.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        subject_train: Subject IDs for training set (for LOSO CV)
        subject_test: Subject IDs for test set
        
    Returns:
        model: Trained LightGBM model
        metrics: Dictionary containing performance metrics
    """
    # Initialize LightGBM classifier with specified parameters
    model = lgb.LGBMClassifier(
        n_estimators=100,
        random_state=42,
        learning_rate=0.1,
        boosting_type='gbdt',
        objective='multiclass',
        num_class=6
    )
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'macro_f1': f1_score(y_test, y_pred, average='macro'),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred, target_names=ACTIVITY_LABELS)
    }
    
    # If subject IDs are provided, perform LOSO cross-validation
    if subject_train is not None and subject_test is not None:
        cv_scores = perform_loso_cv(model, X_train, y_train, subject_train)
        metrics['cv_scores'] = cv_scores
    
    return model, metrics

def perform_loso_cv(
    model: lgb.LGBMClassifier,
    X: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray
) -> Dict[str, list]:
    """
    Perform Leave-One-Subject-Out cross-validation.
    
    Args:
        model: LightGBM classifier
        X: Feature matrix
        y: Labels
        subjects: Subject IDs
        
    Returns:
        Dictionary containing CV scores
    """
    logo = LeaveOneGroupOut()
    cv_accuracies = []
    cv_f1_scores = []
    cv_confusion_matrices = []
    
    for train_idx, val_idx in logo.split(X, y, subjects):
        X_train_cv = X[train_idx]
        y_train_cv = y[train_idx]
        X_val_cv = X[val_idx]
        y_val_cv = y[val_idx]
        
        # Create a new instance of the model for each fold
        model_cv = lgb.LGBMClassifier(**model.get_params())
        
        # Train and evaluate
        model_cv.fit(X_train_cv, y_train_cv)
        y_pred_cv = model_cv.predict(X_val_cv)
        
        # Calculate metrics
        cv_accuracies.append(accuracy_score(y_val_cv, y_pred_cv))
        cv_f1_scores.append(f1_score(y_val_cv, y_pred_cv, average='macro'))
        cv_confusion_matrices.append(confusion_matrix(y_val_cv, y_pred_cv))
    
    return {
        'accuracies': cv_accuracies,
        'f1_scores': cv_f1_scores,
        'confusion_matrices': cv_confusion_matrices,
        'mean_accuracy': np.mean(cv_accuracies),
        'std_accuracy': np.std(cv_accuracies),
        'mean_f1': np.mean(cv_f1_scores),
        'std_f1': np.std(cv_f1_scores)
    }

def plot_confusion_matrix(
    conf_matrix: np.ndarray,
    title: str,
    class_names: List[str] = None,
    save_path: str = None
):
    """
    Plot confusion matrix.
    
    Args:
        conf_matrix: Confusion matrix to plot
        title: Title for the plot
        class_names: List of class names for axis labels
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt='.0f',
        cmap='Blues',
        xticklabels=class_names if class_names else 'auto',
        yticklabels=class_names if class_names else 'auto'
    )
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_performance_metrics(cv_scores: Dict[str, list], save_path: str = None):
    """
    Plot cross-validation performance metrics.
    
    Args:
        cv_scores: Dictionary containing CV scores
        save_path: Path to save the plot (optional)
    """
    plt.figure(figsize=(12, 6))
    
    # Plot accuracy and F1 scores
    plt.subplot(1, 2, 1)
    plt.boxplot([cv_scores['accuracies'], cv_scores['f1_scores']], 
                labels=['Accuracy', 'F1 Score'])
    plt.title('Cross-validation Performance')
    plt.ylabel('Score')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show() 