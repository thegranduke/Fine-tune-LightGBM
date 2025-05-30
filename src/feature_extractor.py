import numpy as np
from scipy.fft import fft
from typing import List, Dict, Union, Tuple

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
        
        # Mean frequency - handle zero weights
        freq_weights = np.arange(len(fft_magnitude))
        if np.sum(fft_magnitude) > 0:
            features[f'mean_freq_{i}'] = np.average(freq_weights, weights=fft_magnitude)
            # Frequency variance
            features[f'freq_variance_{i}'] = np.average(
                np.square(freq_weights - features[f'mean_freq_{i}']),
                weights=fft_magnitude
            )
        else:
            # If all magnitudes are zero, set mean and variance to zero
            features[f'mean_freq_{i}'] = 0
            features[f'freq_variance_{i}'] = 0
    
    return features

def extract_features(X_raw: np.ndarray, return_feature_names: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, List[str]]]:
    """
    Extract time and frequency domain features from raw sensor data.
    
    Args:
        X_raw: Raw sensor data of shape (n_samples, n_timesteps, n_channels)
        return_feature_names: Whether to return feature names along with features
        
    Returns:
        If return_feature_names is False:
            Features array of shape (n_samples, n_features)
        If return_feature_names is True:
            Tuple of (features array, list of feature names)
    """
    n_samples, n_timesteps, n_channels = X_raw.shape
    feature_names = []
    
    # Initialize list to store all features
    all_features = []
    
    # Process each channel
    for channel in range(n_channels):
        channel_data = X_raw[:, :, channel]
        channel_name = f'channel_{channel}'
        
        # Time domain features
        mean = np.mean(channel_data, axis=1)
        std = np.std(channel_data, axis=1)
        rms = np.sqrt(np.mean(np.square(channel_data), axis=1))
        min_val = np.min(channel_data, axis=1)
        max_val = np.max(channel_data, axis=1)
        
        all_features.extend([mean[:, np.newaxis], 
                           std[:, np.newaxis],
                           rms[:, np.newaxis],
                           min_val[:, np.newaxis],
                           max_val[:, np.newaxis]])
        
        if return_feature_names:
            feature_names.extend([
                f'{channel_name}_mean',
                f'{channel_name}_std',
                f'{channel_name}_rms',
                f'{channel_name}_min',
                f'{channel_name}_max'
            ])
        
        # Frequency domain features
        fft_vals = np.fft.fft(channel_data, axis=1)
        fft_freqs = np.fft.fftfreq(n_timesteps)
        
        # Compute spectral energy
        spectral_energy = np.sum(np.abs(fft_vals[:, 1:]) ** 2, axis=1) / n_timesteps
        
        # Find dominant frequency
        dom_freq_idx = np.argmax(np.abs(fft_vals[:, 1:]), axis=1) + 1
        dom_freq = np.abs(fft_freqs[dom_freq_idx])
        
        # Compute mean frequency
        freq_magnitudes = np.abs(fft_vals[:, 1:])
        freq_vals = np.abs(fft_freqs[1:])
        mean_freq = np.sum(freq_magnitudes * freq_vals, axis=1) / (np.sum(freq_magnitudes, axis=1) + 1e-10)
        
        # Compute frequency variance
        freq_var = np.sum(freq_magnitudes * (freq_vals - mean_freq[:, np.newaxis])**2, axis=1) / (np.sum(freq_magnitudes, axis=1) + 1e-10)
        
        all_features.extend([
            spectral_energy[:, np.newaxis],
            dom_freq[:, np.newaxis],
            mean_freq[:, np.newaxis],
            freq_var[:, np.newaxis]
        ])
        
        if return_feature_names:
            feature_names.extend([
                f'{channel_name}_spectral_energy',
                f'{channel_name}_dominant_freq',
                f'{channel_name}_mean_freq',
                f'{channel_name}_freq_variance'
            ])
    
    # Combine all features
    X_features = np.hstack(all_features)
    
    if return_feature_names:
        return X_features, feature_names
    return X_features

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