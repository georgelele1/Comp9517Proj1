from glob import glob

rgb_paths  = sorted(glob(r"C:\Users\28600\Desktop\COMP9517\Segementation_image\USA_segmentation\RGB_images\*.png"))
nrg_paths  = sorted(glob(r"C:\Users\28600\Desktop\COMP9517\Segementation_image\USA_segmentation\NRG_images\*.png"))
mask_paths = sorted(glob(r"C:\Users\28600\Desktop\COMP9517\Segementation_image\USA_segmentation\masks\*.png"))
