from glob import glob

rgb_paths  = sorted(glob("./USA_segmentation/RGB_images/*.png"))
nrg_paths  = sorted(glob("./USA_segmentation/NRG_images/*.png"))
mask_paths = sorted(glob("./USA_segmentation/masks/*.png"))
