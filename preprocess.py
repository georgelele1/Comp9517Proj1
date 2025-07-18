# preprocess.py
import os
from glob import glob
from path import mask_folder,rgb_folder,nrg_folder
import pickle

mask_files = sorted(glob(os.path.join(mask_folder, 'mask_*.png')))
samples = []

for mask_path in mask_files:
    basename = os.path.basename(mask_path)[5:]  # Remove 'mask_'
    rgb_path = os.path.join(rgb_folder, f"RGB_{basename}")
    nrg_path = os.path.join(nrg_folder, f"NRG_{basename}")
    if os.path.exists(rgb_path) and os.path.exists(nrg_path):
        samples.append((rgb_path, nrg_path, mask_path))
    else:
        print(f"Missing RGB or NRG for: {basename}")

print(f"Total samples paired: {len(samples)}")

# Save to paired_samples.pkl
with open('paired_samples.pkl', 'wb') as f:
    pickle.dump(samples, f)
