import numpy as np
from scipy.fft import fft
from typing import List, Dict

def extract_statistical_features(window: np.ndarray) -> Dict[str, float]:
    """
    Extract statistical features from a window of sensor data.
    
    Args:
        window: Array of shape (window_size, n_features)
        
    Returns:
        Dictionary of statistical features
    """
    features = {}
    
    # Mean for each axis
    features.update({
        f'mean_{i}': np.mean(window[:, i]) 
        for i in range(window.shape[1])
    })
    
    # Standard deviation for each axis
    features.update({
        f'std_{i}': np.std(window[:, i]) 
        for i in range(window.shape[1])
    })
    
    # Min for each axis
    features.update({
        f'min_{i}': np.min(window[:, i]) 
        for i in range(window.shape[1])
    })
    
    # Max for each axis
    features.update({
        f'max_{i}': np.max(window[:, i]) 
        for i in range(window.shape[1])
    })
    
    # Root Mean Square (RMS)
    features.update({
        f'rms_{i}': np.sqrt(np.mean(np.square(window[:, i]))) 
        for i in range(window.shape[1])
    })
    
    # Signal Magnitude Area (SMA)
    features['sma_acc'] = np.sum(np.abs(window[:, :3])) / window.shape[0]  # For accelerometer
    features['sma_gyro'] = np.sum(np.abs(window[:, 3:])) / window.shape[0]  # For gyroscope
    
    return features

def extract_spectral_features(window: np.ndarray) -> Dict[str, float]:
    """
    Extract frequency-domain features using FFT.
    
    Args:
        window: Array of shape (window_size, n_features)
        
    Returns:
        Dictionary of spectral features
    """
    features = {}
    
    for i in range(window.shape[1]):
        # Compute FFT
        fft_values = fft(window[:, i])
        fft_magnitude = np.abs(fft_values)[:window.shape[0]//2]
        
        # Spectral energy
        features[f'spectral_energy_{i}'] = np.sum(np.square(fft_magnitude))
        
        # Dominant frequency
        features[f'dominant_freq_{i}'] = np.argmax(fft_magnitude)
        
        # Mean frequency
        freq_weights = np.arange(len(fft_magnitude))
        features[f'mean_freq_{i}'] = np.average(freq_weights, weights=fft_magnitude)
        
        # Frequency variance
        features[f'freq_variance_{i}'] = np.average(
            np.square(freq_weights - features[f'mean_freq_{i}']),
            weights=fft_magnitude
        )
    
    return features

def extract_features(X: np.ndarray) -> np.ndarray:
    """
    Extract all features from windowed data.
    
    Args:
        X: Array of shape (n_windows, window_size, n_features)
        
    Returns:
        Feature matrix of shape (n_windows, n_features)
    """
    n_windows = X.shape[0]
    all_features = []
    
    for i in range(n_windows):
        window = X[i]
        
        # Extract both types of features
        statistical_features = extract_statistical_features(window)
        spectral_features = extract_spectral_features(window)
        
        # Combine all features
        window_features = {**statistical_features, **spectral_features}
        all_features.append(list(window_features.values()))
    
    return np.array(all_features)

def get_feature_names() -> List[str]:
    """
    Get the names of all features in the order they appear in the feature matrix.
    
    Returns:
        List of feature names
    """
    # Create a dummy window to get feature names
    dummy_window = np.zeros((100, 6))  # 100 samples, 6 features (3 acc + 3 gyro)
    
    statistical_features = extract_statistical_features(dummy_window)
    spectral_features = extract_spectral_features(dummy_window)
    
    # Combine all features
    all_features = {**statistical_features, **spectral_features}
    return list(all_features.keys()) 