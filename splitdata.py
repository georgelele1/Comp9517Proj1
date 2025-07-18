import pickle
import random

# Load your paired sample list (produced by preprocess.py)
with open('paired_samples.pkl', 'rb') as f:
    samples = pickle.load(f)

# Shuffle for randomness
random.shuffle(samples)

# 80/20 split for train/test
split_idx = int(0.8 * len(samples))
train_samples = samples[:split_idx]
test_samples  = samples[split_idx:]

print(f"Train samples: {len(train_samples)}")
print(f"Test samples: {len(test_samples)}")

import torch
from torch.utils.data import Dataset
import cv2
import numpy as np

def load_4band_image(rgb_path, nrg_path):
    rgb = cv2.imread(rgb_path, cv2.IMREAD_UNCHANGED)
    nrg = cv2.imread(nrg_path, cv2.IMREAD_UNCHANGED)
    nir = nrg[..., 0:1]  # First channel of NRG is NIR
    four_band = np.concatenate([nir, rgb], axis=-1)  # (H,W,4)
    four_band = four_band.transpose(2, 0, 1) / 255.0  # (4,H,W), normalized
    return four_band.astype(np.float32)

def load_mask(mask_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    mask = (mask > 0).astype(np.float32)
    mask = mask[np.newaxis, ...]  # (1,H,W)
    return mask

class TreeSegmentationDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rgb_path, nrg_path, mask_path = self.samples[idx]
        image = load_4band_image(rgb_path, nrg_path)
        mask = load_mask(mask_path)
        return torch.tensor(image, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)


from torch.utils.data import DataLoader

batch_size = 4  # Or any batch size you prefer

train_ds = TreeSegmentationDataset(train_samples)
test_ds  = TreeSegmentationDataset(test_samples)

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)


