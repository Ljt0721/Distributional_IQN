import numpy as np
import os

files = [
    "pretrained_models/IQN/seed_3/adaptive_evaluations.npz",
    "pretrained_models/DQN/seed_3/evaluations.npz"
]

for f in files:
    if os.path.exists(f):
        print(f"\n--- {f} ---")
        try:
            # allow_pickle=True is needed for some object arrays
            data = np.load(f, allow_pickle=True)
            print(f"Keys: {list(data.files)}")
            for k in data.files:
                val = data[k]
                if isinstance(val, np.ndarray):
                    print(f"  {k}: shape {val.shape}, dtype {val.dtype}")
                    # If it's short, print content
                    if val.size < 10:
                        print(f"    Content: {val}")
        except Exception as e:
            print(f"Error loading {f}: {e}")
