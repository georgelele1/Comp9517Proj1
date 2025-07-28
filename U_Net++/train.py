import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from U_Net import UNetPP
from dataset import SegmentationDataset
from sklearn.model_selection import train_test_split
from paths import rgb_paths, nrg_paths, mask_paths
import pickle
import numpy as np

print("train.py started")

# Device selection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Check consistency of path lists
assert len(rgb_paths) == len(nrg_paths) == len(mask_paths), "Inconsistent number of paths!"

# Split data into training and validation sets (80/20)
train_rgb, val_rgb, train_nrg, val_nrg, train_mask, val_mask = train_test_split(
    rgb_paths, nrg_paths, mask_paths, test_size=0.2, random_state=42
)

# Save validation set indices
test_samples = list(zip(val_rgb, val_nrg, val_mask))
with open('test_samples.pkl', 'wb') as f:
    pickle.dump(test_samples, f)
print("Validation set index has been saved to test_samples.pkl")

# Create datasets and dataloaders
train_dataset = SegmentationDataset(train_rgb, train_nrg, train_mask)
val_dataset = SegmentationDataset(val_rgb, val_nrg, val_mask)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)

# Model setup
model = UNetPP(in_channels=4, out_channels=1).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# --- Loss functions ---
class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=2.0, alpha=0.75):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        inputs = inputs.view(-1)
        targets = targets.view(-1).float()
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce)
        loss = self.alpha * (1 - pt) ** self.gamma * bce
        return loss.mean()

class TverskyLoss(torch.nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, smooth=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1).float()
        tp = (inputs * targets).sum()
        fp = ((1 - targets) * inputs).sum()
        fn = (targets * (1 - inputs)).sum()
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1 - tversky

class DiceLoss(torch.nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1).float()
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        return 1 - dice

# Metric functions
def precision_score(pred, target, threshold=0.5, eps=1e-8):
    pred_bin = (torch.sigmoid(pred) > threshold).float()
    target = target.float()
    tp = (pred_bin * target).sum()
    fp = (pred_bin * (1 - target)).sum()
    return (tp / (tp + fp + eps)).item()

def recall_score(pred, target, threshold=0.5, eps=1e-8):
    pred_bin = (torch.sigmoid(pred) > threshold).float()
    target = target.float()
    tp = (pred_bin * target).sum()
    fn = ((1 - pred_bin) * target).sum()
    return (tp / (tp + fn + eps)).item()

def f1_score(pred, target, threshold=0.5, eps=1e-8):
    prec = precision_score(pred, target, threshold, eps)
    rec = recall_score(pred, target, threshold, eps)
    return 2 * prec * rec / (prec + rec + eps)

focal = FocalLoss()
tversky = TverskyLoss(alpha=0.7, beta=0.3)
dice = DiceLoss()

# IoU calculation function
def iou_score(pred, target, threshold=0.5, smooth=1e-6):
    pred = torch.sigmoid(pred)
    pred_bin = (pred > threshold).float()
    target = target.float()
    intersection = (pred_bin * target).sum()
    union = ((pred_bin + target) > 0).float().sum()
    return ((intersection + smooth) / (union + smooth)).item()

# Training loop
EPOCHS = 100
best_iou = 0.0

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)
        outputs = model(images)
        loss = 0.7 * focal(outputs, masks) + 0.3 * dice(outputs, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
    avg_loss = running_loss / len(train_loader)

    # Validation metrics
    model.eval()
    with torch.no_grad():
        iou_list = []
        f1_list = []
        precision_list = []
        recall_list = []
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            iou_list.append(iou_score(outputs, masks))
            f1_list.append(f1_score(outputs, masks))
            precision_list.append(precision_score(outputs, masks))
            recall_list.append(recall_score(outputs, masks))
        mean_iou = np.mean(iou_list)
        mean_f1 = np.mean(f1_list)
        mean_precision = np.mean(precision_list)
        mean_recall = np.mean(recall_list)

    print(
        f"[Epoch {epoch + 1}/{EPOCHS}] "
        f"Train Loss: {avg_loss:.4f} | "
        f"Val IoU: {mean_iou:.4f} | "
        f"F1: {mean_f1:.4f} | "
        f"Precision: {mean_precision:.4f} | "
        f"Recall: {mean_recall:.4f}"
    )

    if mean_iou > best_iou:
        best_iou = mean_iou
        torch.save(model.state_dict(), "UNetPP_forest_best.pth")
        print(f"Model updated and saved, current best IoU: {best_iou:.4f}")

print("Training finished. Best model saved as UNetPP_forest_best.pth")
