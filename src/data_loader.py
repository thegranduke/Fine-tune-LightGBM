import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List
from sklearn.preprocessing import StandardScaler

def load_data(data_dir: str = "motion-sense/data") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load accelerometer and gyroscope data from the MotionSense dataset.
    
    Args:
        data_dir: Path to the MotionSense data directory
        
    Returns:
        X: Combined sensor data
        y: Activity labels
        subject_ids: Subject identifiers
    """
    data_path = Path(data_dir)
    acc_path = data_path / "B_Accelerometer_data"
    gyro_path = data_path / "C_Gyroscope_data"
    
    if not acc_path.exists() or not gyro_path.exists():
        raise FileNotFoundError(f"Required directories not found in {data_dir}")
    
    all_data = []
    all_labels = []
    all_subjects = []
    
    # Activity mapping
    activity_map = {'dws': 0, 'jog': 1, 'sit': 2, 'std': 3, 'ups': 4, 'wlk': 5}
    
    # Get all activity directories
    acc_dirs = [d for d in acc_path.iterdir() if d.is_dir()]
    
    for acc_dir in acc_dirs:
        # Extract activity and trial info from directory name
        activity, _ = acc_dir.name.split('_')
        if activity not in activity_map:
            continue
            
        activity_idx = activity_map[activity]
        
        # Find matching gyro directory
        gyro_dir = gyro_path / acc_dir.name
        if not gyro_dir.exists():
            continue
            
        # Process all files in these directories
        acc_files = list(acc_dir.glob("sub_*.csv"))
        for acc_file in acc_files:
            # Extract subject ID from filename
            try:
                subject_id = int(acc_file.stem.split('_')[1])
            except (IndexError, ValueError):
                continue
                
            # Find matching gyro file
            gyro_file = gyro_dir / acc_file.name
            if not gyro_file.exists():
                continue
                
            # Read data
            try:
                acc_data = pd.read_csv(acc_file)[['x', 'y', 'z']].values
                gyro_data = pd.read_csv(gyro_file)[['x', 'y', 'z']].values
                
                # Ensure both files have the same length
                min_len = min(len(acc_data), len(gyro_data))
                
                # Combine accelerometer and gyroscope data
                sensor_data = np.hstack([
                    acc_data[:min_len],
                    gyro_data[:min_len]
                ])
                
                all_data.append(sensor_data)
                all_labels.extend([activity_idx] * min_len)
                all_subjects.extend([subject_id] * min_len)
                
            except (pd.errors.EmptyDataError, KeyError) as e:
                print(f"Error reading {acc_file}: {str(e)}")
                continue
    
    if not all_data:
        raise ValueError("No matching accelerometer and gyroscope data found")
        
    return np.vstack(all_data), np.array(all_labels), np.array(all_subjects)

def segment_data(X: np.ndarray, y: np.ndarray, subject_ids: np.ndarray, 
                window_size: int = 100, overlap: float = 0.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Segment data using sliding windows with overlap.
    
    Args:
        X: Sensor data
        y: Activity labels
        subject_ids: Subject identifiers
        window_size: Number of samples in each window
        overlap: Overlap between consecutive windows (0-1)
        
    Returns:
        X_windowed: Segmented sensor data
        y_windowed: Corresponding activity labels
        subjects_windowed: Corresponding subject IDs
    """
    step = int(window_size * (1 - overlap))
    n_samples = X.shape[0]
    n_features = X.shape[1]
    
    # Calculate number of windows
    n_windows = ((n_samples - window_size) // step) + 1
    
    X_windowed = np.zeros((n_windows, window_size, n_features))
    y_windowed = np.zeros(n_windows)
    subjects_windowed = np.zeros(n_windows)
    
    for i in range(n_windows):
        start_idx = i * step
        end_idx = start_idx + window_size
        
        X_windowed[i] = X[start_idx:end_idx]
        # Use majority vote for the label
        y_windowed[i] = np.bincount(y[start_idx:end_idx]).argmax()
        # Use mode for subject ID
        subjects_windowed[i] = np.bincount(subject_ids[start_idx:end_idx]).argmax()
    
    return X_windowed, y_windowed, subjects_windowed

def normalize_per_subject(X: np.ndarray, subject_ids: np.ndarray) -> np.ndarray:
    """
    Apply per-subject normalization to the data.
    
    Args:
        X: Input data
        subject_ids: Subject identifiers
        
    Returns:
        Normalized data
    """
    unique_subjects = np.unique(subject_ids)
    X_normalized = np.zeros_like(X)
    
    for subject in unique_subjects:
        mask = subject_ids == subject
        scaler = StandardScaler()
        
        if len(X.shape) == 3:  # Windowed data
            # Reshape to 2D for scaling
            X_subject = X[mask].reshape(-1, X.shape[-1])
            X_scaled = scaler.fit_transform(X_subject)
            # Reshape back to 3D
            X_normalized[mask] = X_scaled.reshape(-1, X.shape[1], X.shape[2])
        else:  # Raw data
            X_normalized[mask] = scaler.fit_transform(X[mask])
    
    return X_normalized

def prepare_data(data_dir: str = "motion-sense/data", 
                window_size: int = 100,
                overlap: float = 0.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Main function to load, segment, and normalize the data.
    
    Args:
        data_dir: Path to the MotionSense data directory
        window_size: Number of samples in each window
        overlap: Overlap between consecutive windows (0-1)
        
    Returns:
        X: Processed sensor data
        y: Activity labels
        subject_ids: Subject identifiers
    """
    # Load raw data
    X, y, subject_ids = load_data(data_dir)
    
    # Segment data
    X_windowed, y_windowed, subjects_windowed = segment_data(X, y, subject_ids, 
                                                           window_size=window_size,
                                                           overlap=overlap)
    
    # Normalize data per subject
    X_normalized = normalize_per_subject(X_windowed, subjects_windowed)
    
    return X_normalized, y_windowed, subjects_windowed 