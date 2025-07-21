import torch
from torch.utils.data import DataLoader
from U_Net import UNet
from dataset import SegmentationDataset
from sklearn.model_selection import train_test_split
from paths import rgb_paths, nrg_paths, mask_paths
import pickle

print("train.py started")

# Auto-select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Make sure all path lists are matched
assert len(rgb_paths) == len(nrg_paths) == len(mask_paths), "Number of paths do not match!"

# Split into train/validation (80/20)
train_rgb, val_rgb, train_nrg, val_nrg, train_mask, val_mask = train_test_split(
    rgb_paths, nrg_paths, mask_paths, test_size=0.2, random_state=42
)

# Pickle
test_samples = list(zip(val_rgb, val_nrg, val_mask))
with open('test_samples.pkl', 'wb') as f:
    pickle.dump(test_samples, f)
print("Test set index has been saved to test_samples.pkl")

# Create dataset objects
train_dataset = SegmentationDataset(train_rgb, train_nrg, train_mask)
val_dataset   = SegmentationDataset(val_rgb, val_nrg, val_mask)

# Build DataLoader
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=4)

# Create U-Net
model = UNet(in_channels=4, out_channels=1).to(device)

# Loss function
criterion = torch.nn.BCEWithLogitsLoss()

# Adam optimizer, lr=1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Training loop
EPOCHS = 10
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)
        outputs = model(images)
        loss = criterion(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"[Epoch {epoch+1}/{EPOCHS}] Train Loss: {avg_loss:.4f}")

# Save model weights
torch.save(model.state_dict(), "unet_forest.pth")
print("Model has been saved as unet_forest.pth")
