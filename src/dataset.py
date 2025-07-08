import os, glob
from torch.utils.data import Dataset, DataLoader
import cv2
from definitions import *
import torch
import numpy as np

class WheatSegDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        
        # Get all image and mask files
        all_images = sorted([f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
        all_masks = sorted([f for f in os.listdir(masks_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
        
        # Only keep images that have corresponding masks
        self.images = []
        self.masks = []
        
        for img_file in all_images:
            # Get the base name without extension
            base_name = os.path.splitext(img_file)[0]
            
            # Look for corresponding mask file with any common extension
            mask_file = None
            for ext in ['.png', '.jpg', '.jpeg']:
                potential_mask = base_name + ext
                if potential_mask in all_masks:
                    mask_file = potential_mask
                    break
            
            if mask_file:
                self.images.append(img_file)
                self.masks.append(mask_file)
        
        print(f"Found {len(self.images)} matching image-mask pairs in {images_dir}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.images[idx])
        mask_path = os.path.join(self.masks_dir, self.masks[idx])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        image = image.transpose((2, 0, 1))  # HWC to CHW
        image = image / 255.0
        mask = mask / 255.0

        # Ensure positive strides before converting to tensor
        image = np.ascontiguousarray(image)
        mask = np.ascontiguousarray(mask)

        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).float().unsqueeze(0)

        return image, mask

def get_dataloaders(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):
    train_ds = WheatSegDataset("/Users/royayalon/Documents/Academy/final_IP_project/data/train", "/Users/royayalon/Documents/Academy/final_IP_project/data/train_masks")
    val_ds   = WheatSegDataset("/Users/royayalon/Documents/Academy/final_IP_project/data/val",   "/Users/royayalon/Documents/Academy/final_IP_project/data/val_masks")
    return (
      DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers),
      DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    )