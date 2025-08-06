# Comp9517 Project
This project is a computer vision project we have included 4 models Unet Deeplabv3 MasKRcnn and PSP-net each model setup will be following steps 
setup
before you access the model you have to do a setup 
bash 
insteall -r requirements.txt

Dataset preprocess


# PSP Net
## Features

- PSPNet with a ResNet backbone for deep feature extraction.
- Pyramid pooling module to capture global context.
- Skip connections to blend fine details with coarse features.
- Supports training, validation, and visualization of predictions.
- Handles unbalanced datasets with optional loss functions.


## Project Structure

- main.py: Main script to run training or inference.
- config.py: Configuration settings (e.g., USE_NRG, INPUT_CHANNELS).
- utils.py: Helper functions for visualization and metrics.
- model.py: PSPNet model definition.
- dataset.py: Custom dataset loader for NRG_images.
- trainer.py: Training loop and validation logic.
- usa_segmentation\NRG_images: Directory with RGB + NIR image data.
- results/: Folder for saving model outputs and visualizations.

1. Prepare Data

   :

   - Place your images in usa_segmentation\NRG_images (e.g., NRG_wa019_2023_n_33_15_0.png).
   - Ensure masks are in a corresponding directory (e.g., usa_segmentation\masks).
   - Images should be 4-channel (RGB + NIR) if USE_NRG = True, or 3-channel (RGB) if USE_NRG = False.

2. Check Config

   :

   - Edit config.py to set USE_NRG = True (for 4 channels) or False (for 3 channels).
   - Adjust INPUT_CHANNELS and other hyperparameters as needed.

## Usage

Run the project with:

bash

```
python main.py --mode train
```

- --mode train: Starts training with the dataset.
- Output will be saved in results/ (e.g., epoch_001_predictions).

### Training Process

- **Data Loading**: Loads images and masks from usa_segmentation\NRG_images using dataset.py.
- **Model**: PSPNet processes the input with a ResNet backbone and pyramid pooling.
- **Optimization**: Uses Adam optimizer (default lr 0.001) with a scheduler (e.g., ReduceLROnPlateau).
- **Loss**: Default is Cross Entropy Loss; can switch to Dice Loss or Focal Loss for unbalanced data.
- **Visualization**: Generates prediction masks every 10 epochs or at best model, saved as four-panel images (NIR | RGB | Ground Truth | Prediction).



