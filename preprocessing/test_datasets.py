# go over data/preprocessed files and give me dimensions of one instance
import h5py

path = "/home/usirca/workspace/labram-lora/data/preprocessed/ng"

#list all .h5 files in that directory

import os
from pathlib import Path
import h5py

for p in os.listdir(path):
    if p.endswith(".h5"):
        print(p)
        with h5py.File(os.path.join(path, p), "r") as f:
            print("Keys:", list(f.keys()))
            for subject_id in f.keys():
                print(f"Subject {subject_id}:")
                grp = f[subject_id]
                print("  X shape:", grp["X"].shape)
                print("  Y shape:", grp["Y"].shape)
                print("  Sample X[0] shape:", grp["X"][0].shape)
                print("  Sample Y[0]:", grp["Y"][0])
                break  # Just show one subject for brevity
        print("\n")


# for p in os.
# with h5py.File(path, "r") as f:
#     print("Keys:", list(f.keys()))
#     for subject_id in f.keys():
#         print(f"Subject {subject_id}:")
#         grp = f[subject_id]
#         print("  X shape:", grp["X"].shape)
#         print("  Y shape:", grp["Y"].shape)
#         print("  Sample X[0] shape:", grp["X"][0].shape)
#         print("  Sample Y[0]:", grp["Y"][0])
#         break  # Just show one subject for brevity


