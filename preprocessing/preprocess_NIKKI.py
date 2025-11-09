from collections import defaultdict
import numpy as np
import pandas as pd


import h5py
import numpy as np
from scipy.io import loadmat
from scipy.signal import decimate
from tqdm import tqdm
import csv
import io



def csvs_to_h5(input_dir, out_path):
    import os, re, glob, h5py, numpy as np, pandas as pd
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # map filename substring -> trial number offset
    OFFSETS = {"eeg_2": 0, "eeg_3": 40, "eeg_4": 80}
    # We skip Calibration files (because 7 subjects are missing them)

    files = [
        p
        for p in glob.glob(os.path.join(input_dir, "Subject*.csv"))
        if not os.path.basename(p).startswith("._") and "Calibration" not in p
    ]

    # group files by subject id
    subs = {}
    for p in files:
        m = re.search(r"Subject(\d+)", os.path.basename(p))
        if m:
            subs.setdefault(int(m.group(1)), []).append(p)
            
    sid_map = {old_sid: i + 1 for i, old_sid in enumerate(sorted(subs))}


    with h5py.File(out_path, "w") as h5:
        s_int = 1
        for sid in tqdm(sorted(subs)):
            new_sid = sid_map[sid]
            Xs, Ys, Ts = [], [], []
            # process subject files in offset order
            for p in sorted(
                subs[sid],
                key=lambda q: next(
                    (OFFSETS[k] for k in OFFSETS if k in os.path.basename(q)), 9999
                ),
            ):
                off = next((OFFSETS[k] for k in OFFSETS if k in os.path.basename(p)), 0)
                df = pd.read_csv(p)
                cols = [
                    c for c in df.columns if c not in ("TimeStamp", "class", "trial")
                ]
                for i, (_, g) in enumerate(df.groupby("trial"), start=1):
                    g = g[g["TimeStamp"] >= 0]
                    if len(g) == 0:
                        continue
                    g = g.iloc[1:]
                    y = int(g["class"].replace(-1, 0).iloc[0])
                    X = g[cols].to_numpy().T  # (channels, time)
                    Xs.append(X)
                    Ys.append(y)
                    Ts.append(off + i)  # trial ids: 1–40, 41–80, 81–120, 121–160

            grp = h5.create_group(f"s{new_sid}")
            grp.create_dataset(
                "X", data=np.stack(Xs, axis=0)
            )  # (trials, channels, time)
            grp.create_dataset("Y", data=np.asarray(Ys, dtype="i4"))  # (trials,)
            grp.create_dataset("trial_id", data=np.asarray(Ts, dtype="i4"))
            
            s_int += 1


csvs_to_h5("/home/usirca/workspace/labram-lora/data/raw/Leeuwis2021/Raw eeg 2", "/home/usirca/workspace/labram-lora/data/preprocessed/nikki/NIKKI_dataset.h5")
