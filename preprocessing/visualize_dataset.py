import h5py
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import seaborn as sns
from scipy import stats
import math

def visualize_eeg_dataset(dataset_path, subject_id, num_trials=5, class_to_show=None):
    """
    Visualize EEG data for a specific subject and calculate summary statistics
    
    Works with both data formats:
    - (trials, channels, patches, samples)
    - (trials, channels, patches)
    
    Args:
        dataset_path: Path to the HDF5 dataset
        subject_id: Subject ID to visualize
        num_trials: Number of trials to visualize
        class_to_show: Filter trials by class (0 or 1), or None to show all
    """
    # Load data
    with h5py.File(dataset_path, 'r') as f:
        # Check if subject exists
        subject_key = f"s{subject_id}"
        if subject_key not in f:
            print(f"Subject {subject_id} not found in dataset")
            available_subjects = [k for k in f.keys() if k.startswith('s')]
            print(f"Available subjects: {available_subjects}")
            return
            
        # Get data for subject
        X = f[subject_key]['X'][:]  
        Y = f[subject_key]['Y'][:]  
        
        # Print dataset information
        print(f"Dataset shape for subject {subject_id}:")
        print(f"X shape: {X.shape}, Y shape: {Y.shape}")
        
        # Print class distribution
        unique, counts = np.unique(Y, return_counts=True)
        print(f"Class distribution: {dict(zip(unique, counts))}")
        
        # Filter by class if needed
        if class_to_show is not None:
            mask = (Y == class_to_show)
            X = X[mask]
            Y = Y[mask]
            print(f"Selected {X.shape[0]} trials of class {class_to_show}")
        
        # Limit number of trials
        num_trials = min(num_trials, X.shape[0])
        X = X[:num_trials]
        Y = Y[:num_trials]
    
    # Handle different input dimensions
    n_trials = X.shape[0]
    n_channels = X.shape[1]
    
    # Reshape based on dimensionality
    if len(X.shape) == 4:  # (trials, channels, patches, timesteps)
        n_patches, patch_length = X.shape[2], X.shape[3]
        X_reshaped = X.reshape(n_trials, n_channels, n_patches * patch_length)
        print(f"Data format: 4D (trials, channels, patches, timesteps)")
    elif len(X.shape) == 3:  # (trials, channels, patches)
        n_patches = X.shape[2]
        patch_length = 1
        X_reshaped = X.copy()  # No need to reshape
        print(f"Data format: 3D (trials, channels, patches)")
    else:
        raise ValueError(f"Unexpected data shape: {X.shape}. Should be 3D or 4D.")
    
    # Calculate statistics
    mean_per_channel = np.mean(X_reshaped, axis=(0, 2))
    std_per_channel = np.std(X_reshaped, axis=(0, 2))
    min_per_channel = np.min(X_reshaped, axis=(0, 2))
    max_per_channel = np.max(X_reshaped, axis=(0, 2))
    
    # Plot summary statistics
    plt.figure(figsize=(15, 10))
    
    # Plot mean and std for each channel
    plt.subplot(2, 2, 1)
    plt.bar(range(n_channels), mean_per_channel)
    plt.title('Mean value per channel')
    plt.xlabel('Channel index')
    plt.ylabel('Mean value')
    
    plt.subplot(2, 2, 2)
    plt.bar(range(n_channels), std_per_channel)
    plt.title('Standard deviation per channel')
    plt.xlabel('Channel index')
    plt.ylabel('Std dev')
    
    # Plot min/max range per channel
    plt.subplot(2, 2, 3)
    plt.errorbar(range(n_channels), mean_per_channel, 
                 yerr=[mean_per_channel - min_per_channel, max_per_channel - mean_per_channel],
                 fmt='o')
    plt.title('Data range per channel')
    plt.xlabel('Channel index')
    plt.ylabel('Value')
    
    # Plot overall distribution
    plt.subplot(2, 2, 4)
    sns.histplot(X_reshaped.flatten(), kde=True)
    plt.title('Overall signal distribution')
    plt.xlabel('Signal value')
    
    plt.tight_layout()
    plt.savefig(f"subject_{subject_id}_stats.png")
    plt.show()
    
    # Plot trials - with adaptive layout based on channel count
    for trial_idx in range(num_trials):
        # Calculate optimal subplot grid
        # More rows for more channels
        if n_channels <= 16:
            n_rows, n_cols = 4, 4
        elif n_channels <= 32:
            n_rows, n_cols = 6, 6  
        elif n_channels <= 48:
            n_rows, n_cols = 8, 6
        else:  # Up to 64
            n_rows, n_cols = 8, 8
        
        plt.figure(figsize=(15, 12))
        plt.suptitle(f"Subject {subject_id}, Trial {trial_idx}, Class {Y[trial_idx]}")
        
        # Plot each channel - adapt based on dimensionality
        if len(X.shape) == 4:
            # Reshape to combine patches into continuous time series
            trial_data = X[trial_idx].reshape(n_channels, -1)
        else:  # 3D case
            trial_data = X[trial_idx]
        
        # Plot each channel
        for ch_idx in range(n_channels):
            if ch_idx < n_rows * n_cols:  # Ensure we don't exceed subplot grid
                plt.subplot(n_rows, n_cols, ch_idx+1)
                plt.plot(trial_data[ch_idx])
                plt.title(f"Ch {ch_idx}")
                plt.xticks([])  # Hide x ticks for cleaner plot
                
        plt.tight_layout()
        plt.savefig(f"subject_{subject_id}_trial_{trial_idx}_class_{Y[trial_idx]}.png")
        plt.show()
    
    print("\nSummary Statistics:")
    print(f"Global mean: {np.mean(X_reshaped):.4f}")
    print(f"Global std: {np.std(X_reshaped):.4f}")
    print(f"Global min: {np.min(X_reshaped):.4f}")
    print(f"Global max: {np.max(X_reshaped):.4f}")
    
    # Calculate skewness and kurtosis
    skewness = stats.skew(X_reshaped.flatten())
    kurtosis = stats.kurtosis(X_reshaped.flatten())
    print(f"Skewness: {skewness:.4f}")
    print(f"Kurtosis: {kurtosis:.4f}")
    
    # Check for NaNs and Infs
    nan_count = np.isnan(X_reshaped).sum()
    inf_count = np.isinf(X_reshaped).sum()
    print(f"NaN values: {nan_count}")
    print(f"Infinite values: {inf_count}")
    
    # Add data range report to help identify normalization issues
    data_range = np.max(X_reshaped) - np.min(X_reshaped)
    print(f"Data range: {data_range:.8f}")
    if data_range < 0.001:
        print("WARNING: Data has very small range - likely normalized or flattened!")
        print("Check preprocessing for normalization steps.")
    
    return X, Y

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize EEG dataset')
    parser.add_argument('--dataset', type=str, default='/home/usirca/workspace/labram-lora/data/preprocessed/nikki/NIKKI_dataset.h5',
                        help='Path to the HDF5 dataset')
    parser.add_argument('--subject', type=int, default=10, 
                        help='Subject ID to visualize')
    parser.add_argument('--trials', type=int, default=3,
                        help='Number of trials to visualize')
    parser.add_argument('--class_filter', type=int, default=None,
                        help='Class to filter trials (0 or 1, or None for all)')
    
    args = parser.parse_args()
    
    visualize_eeg_dataset(args.dataset, args.subject, args.trials, args.class_filter)