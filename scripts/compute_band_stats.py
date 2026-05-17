"""
Compute per-band mean and std from the GalaxiesML training split.

Run from the project root:
    python scripts/compute_band_stats.py

Output is printed in a format ready to paste into src/data_loader.py.
"""

import h5py
import numpy as np

TRAIN_PATH = "data/5x127x127_training_with_morphology.hdf5"
CHUNK      = 512

with h5py.File(TRAIN_PATH, "r") as f:
    flags   = f["specz_flag_homogeneous"][:].astype(bool)
    indices = np.where(flags)[0]
    images  = f["image"]

    # Pass 1 — mean
    pixel_sum   = np.zeros(5, dtype=np.float64)
    pixel_count = 0
    for start in range(0, len(indices), CHUNK):
        batch = images[np.sort(indices[start:start + CHUNK])].astype(np.float64)
        pixel_sum   += batch.sum(axis=(0, 2, 3))
        pixel_count += batch.shape[0] * 127 * 127
    mean = pixel_sum / pixel_count

    # Pass 2 — std
    pixel_sq_sum = np.zeros(5, dtype=np.float64)
    for start in range(0, len(indices), CHUNK):
        batch = images[np.sort(indices[start:start + CHUNK])].astype(np.float64)
        pixel_sq_sum += ((batch - mean.reshape(1, 5, 1, 1)) ** 2).sum(axis=(0, 2, 3))
    std = np.maximum(np.sqrt(pixel_sq_sum / pixel_count), 1e-8)

print(f"Samples : {len(indices):,}")
print(f"Pixels  : {pixel_count:,}")
print()
print("# Paste into src/data_loader.py:")
print(f"BAND_MEAN = torch.tensor({[round(v,5) for v in mean.tolist()]}, dtype=torch.float32).view(-1,1,1)")
print(f"BAND_STD  = torch.tensor({[round(v,5) for v in std.tolist()]},  dtype=torch.float32).view(-1,1,1)")