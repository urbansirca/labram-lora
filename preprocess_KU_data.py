"""Preprocessor for KU Data with configurable filtering options matching Lee et al. preprocessing."""

from os.path import join as pjoin
from dataclasses import dataclass, field
from typing import List, Dict, Any

import h5py
import numpy as np
from scipy.io import loadmat
from scipy.signal import decimate, butter, filtfilt, iirnotch
from tqdm import tqdm




def get_standard_1020_channels():
    """Get the standard 10-20 EEG channel layout from Lee et al."""
    return [
        "FP1",
        "FPZ",
        "FP2",
        "AF9",
        "AF7",
        "AF5",
        "AF3",
        "AF1",
        "AFZ",
        "AF2",
        "AF4",
        "AF6",
        "AF8",
        "AF10",
        "F9",
        "F7",
        "F5",
        "F3",
        "F1",
        "FZ",
        "F2",
        "F4",
        "F6",
        "F8",
        "F10",
        "FT9",
        "FT7",
        "FC5",
        "FC3",
        "FC1",
        "FCZ",
        "FC2",
        "FC4",
        "FC6",
        "FT8",
        "FT10",
        "T9",
        "T7",
        "C5",
        "C3",
        "C1",
        "CZ",
        "C2",
        "C4",
        "C6",
        "T8",
        "T10",
        "TP9",
        "TP7",
        "CP5",
        "CP3",
        "CP1",
        "CPZ",
        "CP2",
        "CP4",
        "CP6",
        "TP8",
        "TP10",
        "P9",
        "P7",
        "P5",
        "P3",
        "P1",
        "PZ",
        "P2",
        "P4",
        "P6",
        "P8",
        "P10",
        "PO9",
        "PO7",
        "PO5",
        "PO3",
        "PO1",
        "POZ",
        "PO2",
        "PO4",
        "PO6",
        "PO8",
        "PO10",
        "O1",
        "OZ",
        "O2",
        "O9",
        "CB1",
        "CB2",
        "IZ",
        "O10",
        "T3",
        "T5",
        "T4",
        "T6",
        "M1",
        "M2",
        "A1",
        "A2",
        "CFC1",
        "CFC2",
        "CFC3",
        "CFC4",
        "CFC5",
        "CFC6",
        "CFC7",
        "CFC8",
        "CCP1",
        "CCP2",
        "CCP3",
        "CCP4",
        "CCP5",
        "CCP6",
        "CCP7",
        "CCP8",
        "T1",
        "T2",
        "FTT9h",
        "TTP7h",
        "TPP9h",
        "FTT10h",
        "TPP8h",
        "TPP10h",
        "FP1-F7",
        "F7-T7",
        "T7-P7",
        "P7-O1",
        "FP2-F8",
        "F8-T8",
        "T8-P8",
        "P8-O2",
        "FP1-F3",
        "F3-C3",
        "C3-P3",
        "P3-O1",
        "FP2-F4",
        "F4-C4",
        "C4-P4",
        "P4-O2",
    ]


def get_ku_dataset_channels():
    """Get the channel names from the KU dataset."""
    return [
        "Fp1",
        "Fp2",
        "F7",
        "F3",
        "Fz",
        "F4",
        "F8",
        "FC5",
        "FC1",
        "FC2",
        "FC6",
        "T7",
        "C3",
        "Cz",
        "C4",
        "T8",
        "TP9",
        "CP5",
        "CP1",
        "CP2",
        "CP6",
        "TP10",
        "P7",
        "P3",
        "Pz",
        "P4",
        "P8",
        "PO9",
        "O1",
        "Oz",
        "O2",
        "PO10",
        "FC3",
        "FC4",
        "C5",
        "C1",
        "C2",
        "C6",
        "CP3",
        "CPz",
        "CP4",
        "P1",
        "P2",
        "POz",
        "FT9",
        "FTT9h",
        "TTP7h",
        "TP7",
        "TPP9h",
        "FT10",
        "FTT10h",
        "TPP8h",
        "TP8",
        "TPP10h",
        "F9",
        "F10",
        "AF7",
        "AF3",
        "AF4",
        "AF8",
        "PO3",
        "PO4",
    ]


@dataclass
class DataPreprocessingConfig:
    """Configuration class for EEG data preprocessing."""

    # I/O settings
    source_path: str
    target_path: str
    output_name: str = "KU_mi_preprocessed.h5"

    # Sampling parameters
    original_fs: int = 1000
    downsample_factor: int = 5

    # Bandpass filter parameters
    apply_bandpass: bool = True
    bandpass_low: float = 0.5
    bandpass_high: float = 100.0
    bandpass_order: int = 4

    # Notch filter parameters
    apply_notch: bool = True
    notch_frequencies: List[float] = field(default_factory=lambda: [50.0, 60.0, 100.0])
    notch_quality_factor: float = 30.0

    # Common Average Rereferencing
    apply_car: bool = True

    # Patching parameters
    apply_patching: bool = True
    patch_size: int = 1000
    patch_overlap: int = 0

    # Channel selection parameters
    apply_channel_selection: bool = True
    selected_channels: List[str] = field(
        default_factory=list
    )  # Empty means use standard_1020

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.original_fs <= 0:
            raise ValueError("original_fs must be positive")

        if self.downsample_factor <= 0:
            raise ValueError("downsample_factor must be positive")

        if self.apply_bandpass:
            nyquist = self.original_fs / 2
            if not (0 < self.bandpass_low < self.bandpass_high < nyquist):
                raise ValueError(
                    f"Invalid bandpass range: {self.bandpass_low}-{self.bandpass_high} Hz for fs={self.original_fs}"
                )

        if self.apply_notch and not self.notch_frequencies:
            raise ValueError("notch_frequencies cannot be empty when apply_notch=True")

        # Set default channel selection to standard 10-20 if empty and selection is enabled
        if self.apply_channel_selection and not self.selected_channels:
            self.selected_channels = get_standard_1020_channels()

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "DataPreprocessingConfig":
        """Create config from dictionary."""
        return cls(**config_dict)

    @classmethod
    def get_preset(
        cls, preset_name: str, source_path: str, target_path: str
    ) -> "DataPreprocessingConfig":
        """Get preset configuration for common preprocessing pipelines."""
        presets = {
            "labram": {
                "source_path": source_path,
                "target_path": target_path,
                "output_name": "KU_mi_labram_preprocessed.h5",
                "apply_bandpass": True,
                "bandpass_low": 0.5,
                "bandpass_high": 100.0,
                "apply_notch": True,
                "notch_frequencies": [50.0, 60.0, 100.0],
                "apply_car": True,
                "original_fs": 1000,
                "downsample_factor": 5,  # 1000 -> 200 Hz
                "apply_channel_selection": True,
                "apply_patching": True,
                "patch_size": 200,
                "patch_overlap": 0,
            },
            "neurogpt": {
                "source_path": source_path,
                "target_path": target_path,
                "output_name": "KU_mi_neurogpt_preprocessed.h5",
                "apply_bandpass": True,
                "bandpass_low": 0.05,
                "bandpass_high": 100.0,
                "apply_notch": True,
                "notch_frequencies": [50.0, 60.0],
                "apply_car": True,
                "original_fs": 1000,
                "downsample_factor": 4,  # 1000 -> 250 Hz
                "apply_channel_selection": True,
            },
            "minimal": {
                "source_path": source_path,
                "target_path": target_path,
                "output_name": "KU_mi_minimal.h5",
                "apply_bandpass": False,
                "apply_notch": False,
                "apply_car": False,
                "original_fs": 1000,
                "downsample_factor": 4,  # Original script behavior
                "apply_channel_selection": False,
            },
        }

        if preset_name not in presets:
            raise ValueError(
                f"Unknown preset: {preset_name}. Available: {list(presets.keys())}"
            )

        return cls.from_dict(presets[preset_name])

    def get_target_fs(self) -> int:
        """Calculate target sampling frequency after downsampling."""
        return self.original_fs // self.downsample_factor

    def print_summary(self):
        """Print configuration summary."""
        print("Preprocessing configuration:")
        print(f"  Source: {self.source_path}")
        print(f"  Target: {pjoin(self.target_path, self.output_name)}")
        print(f"  Bandpass filter: {self.apply_bandpass}")
        if self.apply_bandpass:
            print(f"    Range: {self.bandpass_low}-{self.bandpass_high} Hz")
        print(f"  Notch filter: {self.apply_notch}")
        if self.apply_notch:
            print(f"    Frequencies: {self.notch_frequencies} Hz")
        print(f"  CAR: {self.apply_car}")
        print(f"  Channel selection: {self.apply_channel_selection}")
        if self.apply_channel_selection:
            print(f"    Target channels: {len(self.selected_channels)} channels")
        print(
            f"  Downsampling: {self.original_fs}/{self.downsample_factor} = {self.get_target_fs()} Hz"
        )


def apply_bandpass_filter(data, lowcut, highcut, fs, order=4):
    """Apply bandpass filter to EEG data."""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, data, axis=-1)


def apply_notch_filter(data, notch_freq, fs, quality_factor=30):
    """Apply notch filter to remove powerline noise."""
    b, a = iirnotch(notch_freq, quality_factor, fs)
    return filtfilt(b, a, data, axis=-1)


def apply_car(data):
    """Apply Common Average Rereferencing across channels.

    Args:
        data: EEG data with shape (epoch, channel, time)

    Returns:
        CAR-referenced data with same shape
    """
    car_reference = np.mean(data, axis=1, keepdims=True)
    return data - car_reference


def select_channels(data, available_channels, target_channels):
    """Select subset of channels from EEG data.

    Args:
        data: EEG data with shape (epoch, channel, time)
        available_channels: List of channel names in the data
        target_channels: List of desired channel names

    Returns:
        tuple: (filtered_data, included_channels, excluded_channels, channel_indices)
    """
    # Normalize channel names to uppercase for matching
    available_upper = [ch.upper() for ch in available_channels]
    target_upper = [ch.upper() for ch in target_channels]

    # Find matching channels
    channel_indices = []
    included_channels = []
    excluded_channels = []

    for i, ch in enumerate(available_upper):
        if ch in target_upper:
            channel_indices.append(i)
            included_channels.append(available_channels[i])
        else:
            excluded_channels.append(available_channels[i])

    # Filter data
    if channel_indices:
        filtered_data = data[:, channel_indices, :]
    else:
        raise ValueError(
            "No matching channels found between dataset and target channels"
        )

    return filtered_data, included_channels, excluded_channels



def create_patches(data, patch_size, overlap=0, verbose=False):
    """
    Convert EEG data to patches for LaBraM-style models.

    Args:
        data: EEG data with shape (epoch, channel, time)
        patch_size: Size of each temporal patch in samples
        overlap: Overlap between patches in samples
        verbose: Whether to print patching info

    Returns:
        patched_data: EEG data with shape (epoch, channel, num_patches, patch_size)
    """
    epochs, channels, time_points = data.shape


    if overlap == 0:
        assert time_points % patch_size == 0, "Time points must be divisible by patch size"
        # Non-overlapping patches
        num_patches = time_points // patch_size
        # Reshape to patches
        patched_data = data.reshape(epochs, channels, num_patches, patch_size)
    else:
        # Overlapping patches
        step_size = patch_size - overlap
        num_patches = max(1, (time_points - overlap) // step_size)
        patched_data = np.zeros(
            (epochs, channels, num_patches, patch_size), dtype=data.dtype
        )

        for i in range(num_patches):
            start_idx = i * step_size
            end_idx = start_idx + patch_size
            if end_idx <= time_points:
                patched_data[:, :, i, :] = data[:, :, start_idx:end_idx]
            else:
                # Handle the last patch if it extends beyond data
                remaining = time_points - start_idx
                patched_data[:, :, i, :remaining] = data[:, :, start_idx:]
                # The rest remains zero-padded

    if verbose:
        print(
            f"  Patching applied: {time_points} -> {num_patches} patches of {patch_size} samples each"
        )

    return patched_data


def get_data(sess, subj, config: DataPreprocessingConfig):
    """Load and preprocess data for one session and subject."""

    filename = "sess{:02d}_subj{:02d}_EEG_MI.mat".format(sess, subj)
    filepath = pjoin(config.source_path, filename)
    raw = loadmat(filepath)

    # Obtain input, convert (time, epoch, chan) into (epoch, chan, time)
    X1 = np.moveaxis(raw["EEG_MI_train"]["smt"][0][0], 0, -1)
    X2 = np.moveaxis(raw["EEG_MI_test"]["smt"][0][0], 0, -1)
    X = np.concatenate((X1, X2), axis=0)

    # Apply channel selection if requested
    if config.apply_channel_selection:
        available_channels = get_ku_dataset_channels()
        target_channels = get_standard_1020_channels()
        X, included_channels, excluded_channels = select_channels(
            X, available_channels, target_channels
        )
        if sess == 1 and subj == 1:
            print(f"\nChannel selection summary:")
            print(f"  Included channels ({len(included_channels)}): {included_channels}")
            print(f"  Excluded channels ({len(excluded_channels)}): {excluded_channels}")

    # Apply bandpass filter before decimation if requested
    if config.apply_bandpass:
        X = apply_bandpass_filter(
            X,
            config.bandpass_low,
            config.bandpass_high,
            config.original_fs,
            order=config.bandpass_order,
        )

    # Apply notch filters if requested
    if config.apply_notch:
        for freq in config.notch_frequencies:
            X = apply_notch_filter(
                X, freq, config.original_fs, quality_factor=config.notch_quality_factor
            )

    # Downsample
    if config.downsample_factor > 1:
        X = decimate(X, config.downsample_factor, axis=-1)

    # Apply Common Average Rereferencing if requested
    if config.apply_car:
        X = apply_car(X)

    # Apply patching if requested
    if config.apply_patching:
        # Only show verbose output for first subject to avoid repetitive printing
        verbose = sess == 1 and subj == 1
        X = create_patches(X, config.patch_size, config.patch_overlap, verbose=verbose)
        if verbose:
            print(f"  Data shape after patching: {X.shape}")

    # Obtain target: 0 -> right, 1 -> left
    Y1 = raw["EEG_MI_train"]["y_dec"][0][0][0] - 1
    Y2 = raw["EEG_MI_test"]["y_dec"][0][0][0] - 1
    Y = np.concatenate((Y1, Y2), axis=0)

    return X, Y


def preprocess_ku_data(config: DataPreprocessingConfig):
    """Main preprocessing function."""

    config.print_summary()

    # Process all subjects
    output_path = pjoin(config.target_path, config.output_name)
    with h5py.File(output_path, "w") as f:
        for subj in tqdm(range(1, 55), desc="Processing subjects"):
            X1, Y1 = get_data(1, subj, config)
            X2, Y2 = get_data(2, subj, config)

            X = np.concatenate((X1, X2), axis=0)
            X = X.astype(np.float32)
            Y = np.concatenate((Y1, Y2), axis=0)
            Y = Y.astype(np.int64)

            f.create_dataset("s" + str(subj) + "/X", data=X)
            f.create_dataset("s" + str(subj) + "/Y", data=Y)

    print(f"Preprocessing complete! Output saved to: {output_path}")


def main():
    source_path = "data/KU_raw"
    target_path = "data/preprocessed"

    # Using preset
    config = DataPreprocessingConfig.get_preset("labram", source_path, target_path)
    preprocess_ku_data(config)


if __name__ == "__main__":
    main()
