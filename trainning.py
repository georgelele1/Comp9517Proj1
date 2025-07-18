import os
import pickle
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torch.nn as nn
import torch.optim as optim
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_4band_image(rgb_path, nrg_path, size=(256, 256)):
    rgb = cv2.imread(rgb_path, cv2.IMREAD_UNCHANGED)
    nrg = cv2.imread(nrg_path, cv2.IMREAD_UNCHANGED)
    nir = nrg[..., 0:1]  # First channel is NIR
    four_band = np.concatenate([nir, rgb], axis=-1)
    # Resize to fixed size
    four_band = cv2.resize(four_band, size, interpolation=cv2.INTER_LINEAR)
    four_band = four_band.transpose(2, 0, 1) / 255.0  # (4, H, W)
    return four_band.astype(np.float32)

def load_mask(mask_path, size=(256, 256)):
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    mask = (mask > 0).astype(np.float32)
    mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)  # Use nearest for masks!
    mask = mask[np.newaxis, ...]
    return mask


class TreeSegmentationDataset(Dataset):
    def __init__(self, samples, size=(256, 256)):
        self.samples = samples
        self.size = size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rgb_path, nrg_path, mask_path = self.samples[idx]
        image = load_4band_image(rgb_path, nrg_path, size=self.size)
        mask = load_mask(mask_path, size=self.size)
        return torch.tensor(image, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)


def compute_iou(pred, mask, threshold=0.4):
    pred = (pred > threshold).float()
    intersection = (pred * mask).sum()
    union = ((pred + mask) > 0).float().sum()
    if union == 0: return float('nan')
    return float((intersection / union).cpu().item())


if __name__ == "__main__":
    # === 1. Load paired samples, shuffle, and split ===
    with open('paired_samples.pkl', 'rb') as f:
        samples = pickle.load(f)

    random.shuffle(samples)
    split_idx = int(0.8 * len(samples))
    train_samples = samples[:split_idx]
    test_samples  = samples[split_idx:]

    print(f"Train samples: {len(train_samples)}")
    print(f"Test samples: {len(test_samples)}")

    # === 2. Dataset ===
    batch_size = 8
    train_ds = TreeSegmentationDataset(train_samples, size=(256, 256))
    test_ds = TreeSegmentationDataset(test_samples, size=(256, 256))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    # === 3. DeepLabV3 Model for 4-Channel Input ===
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torchvision.models.segmentation.deeplabv3_resnet101(weights=None, num_classes=1)
    # Patch as before...
    old_conv = model.backbone.conv1
    model.backbone.conv1 = nn.Conv2d(
        in_channels=4,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=old_conv.bias
    )
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # === 4. Training Loop ===
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)['out']
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss / len(train_loader):.4f}")

    # === 5. Evaluation: Mean IoU ===
    model.eval()
    ious = []
    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)['out']
            outputs = torch.sigmoid(outputs)
            for pred, gt in zip(outputs, masks):
                iou = compute_iou(pred, gt)
                ious.append(iou)
    print(f"Mean IoU on test set: {np.nanmean(ious):.4f}")


    # === 6. (Optional) Visualization ===
    model.eval()
    with torch.no_grad():
        for images, masks in test_loader:
            outputs = model(images.to(device))['out']
            preds = torch.sigmoid(outputs).cpu().numpy()
            imgs = images.numpy()
            msks = masks.numpy()
            break  # Show only first batch

    for i in range(min(4, preds.shape[0])):
        plt.figure(figsize=(16, 3))
        # NIR channel
        plt.subplot(1, 4, 1)
        plt.imshow(imgs[i][0], cmap='gray')
        plt.title('NIR channel')
        # RGB
        plt.subplot(1, 4, 2)
        plt.imshow(imgs[i][1:4].transpose(1, 2, 0))  # RGB
        plt.title('RGB image')
        # Ground truth mask
        plt.subplot(1, 4, 3)
        plt.imshow(msks[i][0], cmap='gray')
        plt.title('Ground Truth')
        # Model prediction
        plt.subplot(1, 4, 4)
        plt.imshow(preds[i][0] > 0.5, cmap='gray')
        plt.title('Prediction')
        plt.savefig(f"result_{i}_with_nir.png")
        plt.close()


