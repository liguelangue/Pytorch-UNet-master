import os
import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
import glob

class MuscleMRIDataset(Dataset):
    def __init__(self, image_dir, mask_dir, split_txt, transform=None, target_size=(256, 256), search_subfolders=True):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.target_size = target_size
        self.search_subfolders = search_subfolders

        # Load image list from split file
        with open(split_txt, 'r') as f:
            self.image_list = [line.strip() for line in f if line.strip()]
        
        # If search_subfolders is True, find all images recursively
        if self.search_subfolders:
            self.image_list = self._find_all_images()

    def _find_all_images(self):
        """Find all PNG images recursively in the image directory"""
        pattern = os.path.join(self.image_dir, '**', '*.png')
        image_files = glob.glob(pattern, recursive=True)
        
        # Convert absolute paths to relative paths from image_dir
        relative_paths = []
        for img_path in image_files:
            rel_path = os.path.relpath(img_path, self.image_dir)
            # Remove .png extension for consistency
            name_without_ext = os.path.splitext(rel_path)[0]
            relative_paths.append(name_without_ext)
        
        return relative_paths

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        name = self.image_list[idx]
        if name.endswith('.png'):
            name = name[:-4]

        # Handle both flat structure and subfolder structure
        if self.search_subfolders:
            # For subfolder structure, name already includes the subfolder path
            img_path = os.path.join(self.image_dir, name + ".png")
            mask_path = os.path.join(self.mask_dir, name + ".png")
        else:
            # For flat structure, use original logic
            img_path = os.path.join(self.image_dir, name + ".png")
            mask_path = os.path.join(self.mask_dir, name + ".png")

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Resize to target size (configurable)
        if self.target_size is not None:
            image = cv2.resize(image, self.target_size, interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, self.target_size, interpolation=cv2.INTER_NEAREST)

        image = image.astype(np.float32) / 255.0
        mask = (mask > 127).astype(np.float32)

        image = np.expand_dims(image, axis=0)
        mask = np.expand_dims(mask, axis=0)

        if self.transform:
            image, mask = self.transform(image, mask)

        return (torch.tensor(image, dtype=torch.float32), 
                torch.tensor(mask, dtype=torch.float32),
                name + ".png")

