import numpy as np
import os

files = ["training_data/evaluations.npz", "training_data/adaptive_evaluations.npz"]
for f in files:
    if os.path.exists(f):
        print(f"--- {f} ---")
        data = np.load(f)
        print(data.files)
        for key in data.files:
            print(f"{key}: {data[key].shape}")
