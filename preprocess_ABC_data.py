import os, re, h5py, numpy as np, mne
from pathlib import Path
from mne.io import read_raw_gdf

EEG_27 = ['Fz','FCz','Cz','CPz','Pz','C1','C3','C5','C2','C4','C6','F4','FC2','FC4','FC6',
          'CP2','CP4','CP6','P4','F3','FC1','FC3','FC5','CP1','CP3','CP5','P3']

CODES = {'TRIAL_START':768, 'CUE_LEFT':769, 'CUE_RIGHT':770, 'FEEDBACK_START':781, 'TRIAL_END':800}

def get_ABC_dataset_channels():
    return [i.upper() for i in EEG_27]

def load_runs(subj_dir: Path):
    pat = re.compile(rf'^{re.escape(subj_dir.name)}_R([1-6])_.*\.gdf$', re.I)
    run_files = []
    for f in subj_dir.glob('*.gdf'):
        m = pat.match(f.name)
        if m: run_files.append((int(m.group(1)), f))
    return [f for _,f in sorted(run_files)]

def preprocess(raw: mne.io.BaseRaw,
               apply_bandpass: bool = True, bandpass_low: float = 0.5, bandpass_high: float = 100.0,
               apply_notch: bool = True, notch_frequencies: list[float] | None = None,
               apply_car: bool = True) -> mne.io.BaseRaw:
    """
    LaBraM-like steps adapted to this dataset:
      - Band-pass (default 0.5-100 Hz; for MI you may switch to 5-35 Hz if desired)
      - Notch at EU mains & 2nd harmonic (50, 100 Hz). No 60 Hz here.
      - Common Average Reference across the 27 EEG channels
    """
    if notch_frequencies is None:
        notch_frequencies = [50.0, 100.0]

    raw.load_data()

    if apply_bandpass:
        raw.filter(l_freq=bandpass_low, h_freq=bandpass_high, verbose=False)

    if apply_notch and len(notch_frequencies) > 0:
        raw.notch_filter(freqs=notch_frequencies, verbose=False)

    if apply_car:
        X = raw.get_data()
        X -= X.mean(axis=0, keepdims=True)  # CAR across channels
        raw._data[:] = X

    return raw

# ---- robust event extraction: stim channel if available, else parse annotations ----
def _get_events_int(raw: mne.io.BaseRaw) -> np.ndarray:

    sf = raw.info['sfreq']
    rows = []
    # Robustly parse all annotations
    for ann in raw.annotations:
        desc = ann['description']
        if isinstance(desc, (bytes, bytearray)):
            desc = desc.decode()
        code = None
        if isinstance(desc, str):
            # (i) direct digits anywhere in the string
            m = re.search(r"(\d+)", desc)
            if m:
                try:
                    code = int(m.group(1))
                    # print(f"  -> Parsed numeric code {code} from digits in string.")
                except ValueError:
                    # print("  -> Failed to convert digits to int.")
                    code = None

        if code is None:
            # print(f"  -> Skipping unknown label: {desc}")
            continue  # skip unknown labels

        sample = int(round((ann['onset'] - raw.first_time) * sf))
        rows.append([sample, 0, code])

    ev = np.array(rows, dtype=int)
    # print(f"[get_events_int] Parsed {len(ev)} events from annotations (before filtering).")

    # Keep only the codes we care about
    keep_codes = [768, 769, 770, 781, 800]
    keep = np.isin(ev[:, 2], keep_codes)
    if keep.any():
        # print(f"[get_events_int] Retaining {keep.sum()} events with known codes {keep_codes}.")
        return ev[keep]
    else:
        # print("[get_events_int] No events with known codes were found.")
        return np.empty((0, 3), dtype=int)


# ---- epoching: fixed window [4.0, 8.0] s from TRIAL_START, no resampling ----
def epochs_from_events(raw: mne.io.BaseRaw,
                       events: np.ndarray,
                       tmin: float = 4.0, tmax: float = 8.0) -> tuple[np.ndarray, np.ndarray]:
    sf = raw.info['sfreq']            # expected 512.0
    data = raw.get_data()             # (C, N)
    if events.size == 0:
        raise RuntimeError("No events found")

    starts = events[events[:,2] == CODES['TRIAL_START']][:,0]
    if starts.size == 0:
        raise RuntimeError("No TRIAL_START events")

    X_list, Y_list = [], []
    N = data.shape[1]
    for i, s0 in enumerate(starts):
        s1 = starts[i+1] if i+1 < len(starts) else N
        mask = (events[:,0] > s0) & (events[:,0] < s1) & np.isin(events[:,2], [CODES['CUE_LEFT'], CODES['CUE_RIGHT']])
        if not mask.any():
            continue
        cue = events[mask][0, 2]
        y = 1 if cue == CODES['CUE_LEFT'] else 0  # 1=left, 0=right

        i0 = int(s0 + tmin * sf)
        i1 = int(s0 + tmax * sf)
        if i0 < 0 or i1 > N or (i1 - i0) <= 0:
            continue

        seg = data[:, i0:i1]  # (C, L_src) = 2048 if 4s @ 512 Hz
        X_list.append(seg.astype('float32'))
        Y_list.append(y)

    if not X_list:
        raise RuntimeError("No valid epochs created")
    X = np.stack(X_list, axis=0)        # (T, C, L)
    Y = np.array(Y_list, dtype='int64') # (T,)
    return X, Y


def pick_27(raw: mne.io.BaseRaw):
    chs = {ch.upper(): ch for ch in raw.ch_names}
    # print(chs)
    missing = [ch for ch in EEG_27 if ch not in chs and ch.upper() not in chs]
    if missing:
        raise RuntimeError(f"Missing EEG channels: {missing}")
    sel = [chs.get(ch, chs[ch.upper()]) for ch in EEG_27]
    # print(raw.annotations)
    return raw.pick(sel, verbose=False)

def to_patches_nonoverlap(X: np.ndarray, patch_size: int = 512) -> np.ndarray:
    """
    X: (T, C, L) where L must be divisible by patch_size
    returns: (T, C, P, patch_size) with P = L // patch_size
    """
    T, C, L = X.shape
    if L % patch_size != 0:
        raise ValueError(f"L={L} not divisible by patch_size={patch_size}")
    P = L // patch_size
    return X.reshape(T, C, P, patch_size)

from fractions import Fraction
import mne

from fractions import Fraction
import numpy as np
import mne

def resample_last_axis_to_len(X: np.ndarray, target_len: int) -> np.ndarray:
    """
    Resample along the last axis (time) to an exact target length using polyphase filtering.
    X: (T, C, L_src)  ->  (T, C, target_len)
    """
    T, C, L_src = X.shape
    if L_src == target_len:
        return X

    # Ensure C-contiguous and float64 for SciPy's resample_poly
    X64 = np.ascontiguousarray(X, dtype=np.float64)

    frac = Fraction(target_len, L_src).limit_denominator()
    up, down = frac.numerator, frac.denominator

    X_rs = mne.filter.resample(X64, up=up, down=down, axis=-1, npad="auto")

    # Guard against off-by-one
    if X_rs.shape[-1] != target_len:
        if X_rs.shape[-1] > target_len:
            X_rs = X_rs[..., :target_len]
        else:
            pad = target_len - X_rs.shape[-1]
            X_rs = np.pad(X_rs, ((0,0),(0,0),(0,pad)), mode='edge')

    return X_rs.astype(np.float32, copy=False)


def convert_subject(subj_dir: Path):
    raws = []
    all_events = []
    cum_offset = 0

    for run_file in load_runs(subj_dir):
        raw_full = read_raw_gdf(run_file, preload=False, verbose='ERROR')

        # 1) grab events BEFORE dropping stim/annotations
        ev = _get_events_int(raw_full)
        if ev.size:
            ev[:,0] += cum_offset           # offset events by previous runs
            all_events.append(ev)
        cum_offset += raw_full.n_times      # advance offset by run length (samples)

        # 2) now keep only the 27 EEG channels and preprocess
        raw_eeg = pick_27(raw_full.copy())
        raw_eeg = preprocess(raw_eeg)
        raws.append(raw_eeg)

    if not raws:
        return None

    raw_cat = mne.concatenate_raws(raws, verbose=False)
    events_cat = np.vstack(all_events) if len(all_events) else np.empty((0,3), dtype=int)

    X, Y = epochs_from_events(raw_cat, events_cat, tmin=4.0, tmax=8.0)  # X: (T, 27, 2048)
    # Resample each 4 s epoch from 2048 -> 800 samples (i.e., 200 Hz) to match LaBraM patch200 checkpoints
    X = resample_last_axis_to_len(X, target_len=800)        

    X = to_patches_nonoverlap(X, patch_size=200)                         # -> (T, 27, 4, 200)
    return X, Y

def write_h5(out_path: Path, subjects: list[Path], scheme='sequential'):
    with h5py.File(out_path, 'w') as h5:
        sid = 1
        for subj in subjects:
            try:
                X, Y = convert_subject(subj)
            except Exception as e:
                print(f"[skip] {subj.name}: {e}")
                continue
            gname = f"s{sid}" if scheme == 'sequential' else subj.name.lower()
            g = h5.create_group(gname)
            g.create_dataset("X", data=X, dtype='float32')
            g.create_dataset("Y", data=Y, dtype='int64')
            print(f"[ok] {subj.name} -> /{gname} | X={X.shape} Y={Y.shape}")
            sid += 1

if __name__ == "__main__":
    root = Path("/home/usirca/workspace/labram-lora/BCI Database/Signals") 

    # Store paths of all subjects (example: BCI_Database/Signals/DATA B/B79)
    subjects = []
    for dset in ["DATA A","DATA B","DATA C"]:
        for s in sorted((root/dset).iterdir(), key=lambda p: (p.name.rstrip('0123456789'), int(re.findall(r'\d+', p.name)[0]))):
            if s.is_dir(): 
                subjects.append(s)
                print(s)

    write_h5(Path("data/preprocessed/ABC_labram_preprocessed.h5"), subjects)
