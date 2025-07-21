print("predict.py started")

import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from U_Net import UNet
from dataset import SegmentationDataset
import pickle
import os

# Config
model_path = "unet_forest.pth"
test_samples_pkl = "test_samples.pkl"
save_dir = "predict_results"
os.makedirs(save_dir, exist_ok=True)

# Device selection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# IoU calculation
def compute_iou(pred, mask, threshold=0.5):
    pred = (pred > threshold).float()
    intersection = (pred * mask).sum()
    union = ((pred + mask) > 0).float().sum()
    if union == 0:
        return float('nan')
    return float((intersection / union).item())

# Load trained model
model = UNet(in_channels=4, out_channels=1).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Load test set samples
with open(test_samples_pkl, 'rb') as f:
    test_samples = pickle.load(f)

test_dataset = SegmentationDataset(
    [x[0] for x in test_samples],
    [x[1] for x in test_samples],
    [x[2] for x in test_samples]
)

num_to_save = 5
ious = []
for idx in range(len(test_dataset)):
    image, mask = test_dataset[idx]
    image = image.unsqueeze(0).to(device)
    mask = mask.unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        output = model(image)
        pred = torch.sigmoid(output)

    # Compute IoU
    iou = compute_iou(pred, mask)
    ious.append(iou)

    # Output
    if idx < num_to_save:
        pred_np = pred.squeeze().cpu().numpy()
        mask_np = mask.squeeze().cpu().numpy()
        img_np  = image.squeeze().cpu().numpy()
        rgb_img = (img_np[1:4].transpose(1, 2, 0) * 255).astype(np.uint8)
        nir_img = (img_np[0] * 255).astype(np.uint8)

        plt.figure(figsize=(10, 2))
        
        plt.subplot(1, 4, 1)
        plt.imshow(nir_img, cmap="gray")
        plt.title("NIR Channel")
        
        plt.subplot(1, 4, 2)
        plt.imshow(rgb_img)
        plt.title("RGB Image")
        
        plt.subplot(1, 4, 3)
        plt.imshow(mask_np, cmap="gray")
        plt.title("Ground Truth Mask")
        
        plt.subplot(1, 4, 4)
        plt.imshow(pred_np > 0.5, cmap="gray")
        plt.title("Predicted Mask")

        plt.tight_layout()
        rgb_name = os.path.basename(test_samples[idx][0]).replace('.png', '')
        out_file = os.path.join(save_dir, f"{idx:03d}_{rgb_name}_result.png")
        plt.savefig(out_file, dpi=200)
        plt.close()

print(f"Top {num_to_save} prediction results saved in {save_dir}")
print(f"Mean IoU on test set: {np.nanmean(ious):.4f}")
