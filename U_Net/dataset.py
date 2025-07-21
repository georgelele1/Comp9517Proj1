import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

# Custom dataset class for forest image segmentation
class SegmentationDataset(Dataset):
    def __init__(self, rgb_paths, nrg_paths, mask_paths, size=(256, 256)):
        self.rgb_paths = rgb_paths
        self.nrg_paths = nrg_paths
        self.mask_paths = mask_paths
        self.size = size

    def __len__(self):
        return len(self.rgb_paths)

    def __getitem__(self, idx):
        # Load RGB and NRG images
        rgb = cv2.imread(self.rgb_paths[idx])
        nrg = cv2.imread(self.nrg_paths[idx])
        if rgb is None or nrg is None:
            raise ValueError(f"Failed to load image: {self.rgb_paths[idx]} or {self.nrg_paths[idx]}")

        # Resize images
        rgb = cv2.resize(rgb, self.size)
        nrg = cv2.resize(nrg, self.size)

        # Extract NIR channel and concatenate with RGB
        nir = nrg[..., 0:1]
        four_band = np.concatenate([nir, rgb], axis=-1)
        four_band = four_band.transpose(2, 0, 1) / 255.0

        # Load and process mask
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_UNCHANGED)
        mask = cv2.resize(mask, self.size, interpolation=cv2.INTER_NEAREST)
        mask = (mask > 0).astype(np.float32)
        mask = mask[np.newaxis, ...]

        # Convert to PyTorch tensors and return
        return torch.tensor(four_band, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)
