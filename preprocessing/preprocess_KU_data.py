"""Preprocessor for KU Data with configurable filtering options matching Lee et al. preprocessing."""

from os.path import join as pjoin
from dataclasses import dataclass, field
from typing import List, Dict, Any

import h5py
import numpy as np
from scipy.io import loadmat
from scipy.signal import decimate, butter, filtfilt, iirnotch
from tqdm import tqdm

def _to_ECT(X):
    """
    Ensure shape is (E, C, T). If X is (E, C, P, S), merge P*S -> T.
    Returns:
        X_ect : (E,C,T)
        info  : dict for reshaping back
    """
    if X.ndim == 3:
        E, C, T = X.shape
        return X, {"patched": False, "shape": (E, C, T)}
    elif X.ndim == 4:
        E, C, P, S = X.shape
        X_ect = X.reshape(E, C, P * S)
        return X_ect, {"patched": True, "shape": (E, C, P, S)}
    else:
        raise ValueError(f"Unsupported X.ndim={X.ndim}; expected 3 or 4")


def _from_ECT(X_ect, info):
    """
    Reshape back to original shape using info from _to_ECT.
    """
    if not info["patched"]:
        return X_ect.reshape(info["shape"])
    else:
        E, C, P, S = info["shape"]
        return X_ect.reshape(E, C, P, S)


# ----------------------------
# Core normalization utilities
# ----------------------------


def _safe_stats(x, axis, eps):
    """
    Return (mean, std) with std floored by eps to avoid divide-by-zero.
    """
    mu = np.nanmean(x, axis=axis, keepdims=True)
    sd = np.nanstd(x, axis=axis, keepdims=True)
    sd = np.maximum(sd, eps)
    return mu, sd


def normalize_per_subject_channel(X, support_mask=None, eps=1e-6):
    """
    Z-score per channel using stats computed over the selected 'support' trials
    of the same subject. Applies those stats to all trials.

    Args:
        X : np.ndarray, shape (E,C,T) or (E,C,P,S)
        support_mask : None or boolean array of shape (E,). If None, use all E as support.
        eps : float, numerical stability

    Returns:
        X_norm : same shape as X
        stats  : dict with 'mu' and 'std' of shape (1,C,1) for ECT, applied to all trials
    """
    X_ect, info = _to_ECT(X)
    E, C, T = X_ect.shape
    if support_mask is None:
        support_mask = np.ones(E, dtype=bool)

    # Compute μ,σ over (support epochs, time) per channel
    x_sup = X_ect[support_mask]  # (Esup, C, T)
    mu, sd = _safe_stats(x_sup, axis=(0, 2), eps=eps)  # (1, C, 1)

    # Apply to all trials
    Xn = (X_ect - mu) / sd
    Xn = _from_ECT(Xn, info)
    stats = {"mu": mu.squeeze(axis=(0, 2)), "std": sd.squeeze(axis=(0, 2))}
    return Xn, stats


def normalize_per_run_channel(X, run_ids=None, support_mask=None, eps=1e-6):
    """
    Z-score per channel, *per run*. Stats are computed within each run over
    support trials (if provided), then applied to all trials in that run.

    Args:
        X : np.ndarray, shape (E,C,T) or (E,C,P,S)
        run_ids : array of shape (E,), values in {0,1,2,3}. If None, assumes 4 runs of 100 trials each.
        support_mask : None or boolean array of shape (E,). If provided, stats per run are computed on support trials within that run.
        eps : float

    Returns:
        X_norm : same shape as X
        stats  : dict with 'mu' and 'std' arrays of shape (R, C)
    """
    X_ect, info = _to_ECT(X)
    E, C, T = X_ect.shape

    # Default run_ids: 4 runs × 100 trials (assert E == 400)
    if run_ids is None:
        if E % 100 != 0:
            raise ValueError(
                "run_ids not provided and E is not a multiple of 100; cannot infer 4 runs × 100."
            )
        R = E // 100
        run_ids = np.repeat(np.arange(R), 100)
    else:
        run_ids = np.asarray(run_ids)
        R = int(run_ids.max()) + 1

    if support_mask is None:
        support_mask = np.ones(E, dtype=bool)

    Xn = np.empty_like(X_ect)
    mu_all = np.zeros((R, C), dtype=X_ect.dtype)
    sd_all = np.zeros((R, C), dtype=X_ect.dtype)

    for r in range(R):
        idx_all = run_ids == r
        idx_sup = idx_all & support_mask
        if not np.any(idx_sup):
            # fallback: if no support in this run, use all trials in the run (still leakage-safe if you design support_mask to include at least some)
            idx_sup = idx_all

        x_sup = X_ect[idx_sup]  # (E_run_sup, C, T)
        mu, sd = _safe_stats(x_sup, axis=(0, 2), eps=eps)  # (1, C, 1)
        mu_all[r] = mu.squeeze()
        sd_all[r] = sd.squeeze()

        # normalize all trials in this run with that run's stats
        Xn[idx_all] = (X_ect[idx_all] - mu) / sd

    Xn = _from_ECT(Xn, info)
    stats = {"mu": mu_all, "std": sd_all}
    return Xn, stats


def normalize_per_trial_channel(X, eps=1e-6):
    """
    Z-score per channel, *per trial* (instance normalization).
    Each trial (epoch) is normalized using its own (mean,std) per channel over time.

    Args:
        X : np.ndarray, shape (E,C,T) or (E,C,P,S)
        eps : float

    Returns:
        X_norm : same shape as X
        stats  : dict with 'mu' and 'std' of shape (E, C)
    """
    X_ect, info = _to_ECT(X)
    E, C, T = X_ect.shape

    # Compute (E,C,1) stats
    mu = np.nanmean(X_ect, axis=2, keepdims=True)  # (E,C,1)
    sd = np.nanstd(X_ect, axis=2, keepdims=True)  # (E,C,1)
    sd = np.maximum(sd, eps)

    Xn = (X_ect - mu) / sd
    Xn = _from_ECT(Xn, info)
    stats = {"mu": mu.squeeze(-1), "std": sd.squeeze(-1)}  # (E,C)
    return Xn, stats


# ----------------------------
# Convenience dispatcher
# ----------------------------


def normalize_X(
    X,
    mode="subject",  # "subject" | "run" | "trial"
    support_mask=None,  # boolean (E,) for "subject" or "run" modes
    run_ids=None,  # array (E,) for "run" mode; if None assumes 4×100
    eps=1e-6,
):
    """
    Dispatch to the chosen normalization strategy.
    Returns (X_norm, stats)
    """
    if mode == "subject":
        return normalize_per_subject_channel(X, support_mask=support_mask, eps=eps)
    elif mode == "run":
        return normalize_per_run_channel(
            X, run_ids=run_ids, support_mask=support_mask, eps=eps
        )
    elif mode == "trial":
        return normalize_per_trial_channel(X, eps=eps)
    else:
        raise ValueError(f"Unknown mode: {mode}")


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
        "FP1",
        "FP2",
        "F7",
        "F3",
        "FZ",
        "F4",
        "F8",
        "FC5",
        "FC1",
        "FC2",
        "FC6",
        "T7",
        "C3",
        "CZ",
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
        "PZ",
        "P4",
        "P8",
        "PO9",
        "O1",
        "OZ",
        "O2",
        "PO10",
        "FC3",
        "FC4",
        "C5",
        "C1",
        "C2",
        "C6",
        "CP3",
        "CPZ",
        "CP4",
        "P1",
        "P2",
        "POZ",
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

def get_nikki_dataset_channels():
    NIKKI_ELECTRODES = ["F3", "Fz", "F4", "FC5", "FC1", "FC2", "FC6", "T7", "C3", "C4", "Cz", "T8", "CP5", "CP1", "CP2", "CP6"]
    return NIKKI_ELECTRODES


def get_dreyer_dataset_channels():
    DREYER_32 = [
    'FZ','FCZ','CZ','CPZ','PZ',
    'C1','C3','C5','C2','C4','C6',
    'F4','FC2','FC4','FC6',
    'CP2','CP4','CP6',
    'P4',
    'F3','FC1','FC3','FC5',
    'CP1','CP3','CP5',
    'P3',
    ]

    # capitalize
    DREYER_32 = [ch.upper() for ch in DREYER_32]
    return DREYER_32

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
    
    # Normalization parameters
    normalize_mode: str = "subject"  # "subject" | "run" | "trial"

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
                "output_name": "KU_mi_labram_preprocessed_trial_normalized.h5",
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
                "normalize_mode": "trial", 
            }
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
        assert (
            time_points % patch_size == 0
        ), "Time points must be divisible by patch size"
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
            print(
                f"  Included channels ({len(included_channels)}): {included_channels}"
            )
            print(
                f"  Excluded channels ({len(excluded_channels)}): {excluded_channels}"
            )

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


def _get_train_count_for_session(source_path: str, sess: int, subj: int) -> int:
    """
    Return number of train trials for (sess, subj) from the raw .mat,
    so we can build a leakage-safe support_mask.
    """
    fname = f"sess{sess:02d}_subj{subj:02d}_EEG_MI.mat"
    raw = loadmat(pjoin(source_path, fname))
    # EEG_MI_train['smt'] has shape (time, epochs, channels); epochs is axis=1
    n_train = raw["EEG_MI_train"]["smt"][0][0].shape[1]
    return int(n_train)


def preprocess_ku_data(config: DataPreprocessingConfig):
    """Main preprocessing function."""

    config.print_summary()

    output_path = pjoin(config.target_path, config.output_name)
    with h5py.File(output_path, "w") as f:
        for subj in tqdm(range(1, 55), desc="Processing subjects"):
            # Get per-session data (each already concatenates train+test inside get_data)
            X1, Y1 = get_data(1, subj, config)
            X2, Y2 = get_data(2, subj, config)

            # Concatenate sessions
            X = np.concatenate((X1, X2), axis=0)
            Y = np.concatenate((Y1, Y2), axis=0)

            # -------------------- NORMALIZATION (drop-in) --------------------
            # Build a leakage-safe support mask: mark TRAIN trials as support, TEST as query
            # We read counts from the raw .mat to avoid guessing.
            n_train_s1 = _get_train_count_for_session(config.source_path, 1, subj)
            n_train_s2 = _get_train_count_for_session(config.source_path, 2, subj)

            E1 = X1.shape[0]  # total trials in session 1 (train+test after get_data)
            E2 = X2.shape[0]  # total trials in session 2
            support_mask = np.zeros(E1 + E2, dtype=bool)
            # Session 1 occupies [0, E1) in X
            support_mask[:n_train_s1] = True
            # Session 2 occupies [E1, E1+E2) in X
            support_mask[E1 : E1 + n_train_s2] = True

            # --- Choose ONE normalization mode ---
            # A) Per-subject, per-channel (recommended for meta-learning; leakage-safe)
            if config.normalize_mode == "subject":
                Xn, stats = normalize_X(
                    X, mode="subject", support_mask=support_mask, eps=1e-6
                )

            # B) Per-run, per-channel (uncomment if you prefer run-wise stats)
            # If you truly have 4 runs × 100 trials after concatenating s1+s2:
            elif config.normalize_mode == "run":
                assert X.shape[0] == 400, "Expected 400 trials for run-wise normalization"
                run_ids = np.repeat(np.arange(4), 100)  # length must equal X.shape[0] (=400)
                Xn, stats = normalize_X(X, mode="run", run_ids=run_ids, support_mask=support_mask, eps=1e-6)

            # C) Per-trial, per-channel (instance normalization; always leakage-safe but removes amplitude cues)
            elif config.normalize_mode == "trial":
                Xn, stats = normalize_X(X, mode="trial", eps=1e-6)
            else:
                raise ValueError(f"Unknown normalization_mode: {config.normalize_mode}")

            # -------------------- SAVE --------------------
            Xn = Xn.astype(np.float32)
            Y = Y.astype(np.int64)

            grp = f.create_group(f"s{subj}")
            grp.create_dataset("X", data=Xn)
            grp.create_dataset("Y", data=Y)

            # also save normalization stats so you can reapply at inference
            # Subject-level stats:
            if "mu" in stats and "std" in stats:
                grp.create_dataset("norm/mu", data=np.asarray(stats["mu"]))
                grp.create_dataset("norm/std", data=np.asarray(stats["std"]))
            # If you used run-wise stats, 'mu'/'std' will be shape (R, C) — still fine to save.

    print(f"Preprocessing complete! Output saved to: {output_path}")


def main():
    source_path = "data/KU_raw"
    target_path = "data/preprocessed"

    # Using preset
    config = DataPreprocessingConfig.get_preset("labram", source_path, target_path)
    preprocess_ku_data(config)


if __name__ == "__main__":
    main()

    # indices = []
    # for channel in get_ku_dataset_channels():
    #     if channel.upper() in get_standard_1020_channels():
    #         indices.append(get_standard_1020_channels().index(channel.upper()))
    # # get the standard 1020 channels with indices
    # standard_1020_channels_with_indices = [get_standard_1020_channels()[i] for i in indices]
    # print(standard_1020_channels_with_indices)
