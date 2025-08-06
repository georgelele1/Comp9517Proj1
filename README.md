# Dataset Preprocessing and Model Training

## 📌 Overview
This project involves preprocessing a dataset of RGB, NIR, and Mask images, pairing them into `.pkl` files, and training segmentation models such as **Mask R-CNN** and **DeepLabv3**.

---

## 🔧 Preprocessing Steps

1. **Count Background and Tree Pixels**  
   Run `count.py` to measure the total number of background pixels and tree pixels in the dataset.

2. **Pair the Dataset**  
   Use `datamatch.py` to pair the data from the three folders (**RGB**, **NIR**, and **Mask**) and generate a `.pkl` file.

3. **Access Model Folders**  
   Navigate into each model’s folder (e.g., `deeplabv3/`, `maskrcnn/`) to run training scripts.

---

## 🚀 Training the Models
Change the PKL file dict inside the train 
Both **Mask R-CNN** and **DeepLabv3** support flexible training parameters.  
Run the training script with desired arguments:

```bash
python train.py --args parameter --args parameter
```
Deeplabv3 to activate the cbam attention by 

```bash 
--use_cbam
```
Predict the model by `predict.py` under each corresponding model folder because each mode may have different output layout thus predict is for each model sepecifialy 
