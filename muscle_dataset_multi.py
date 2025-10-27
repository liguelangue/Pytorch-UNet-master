import os
import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
import glob

class MuscleMRIDatasetMulti(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, target_size=(256, 256), search_subfolders=True, 
                num_classes=3):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.target_size = target_size
        self.search_subfolders = search_subfolders
        self.num_classes = num_classes
        
        # Define color to class mapping based on your analysis
        self.color_to_class = {
            (0, 0, 0): 0,        # Background (black)
            (21, 21, 21): 1,     # Muscle #21 class 1 (gray value 21)
            (90, 90, 90): 2,     # Muscle #90 class 2 (gray value 90)
        }
        
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

    def _color_to_class_mask(self, color_mask):
        """Convert color mask to class indices"""
        h, w, c = color_mask.shape
        class_mask = np.zeros((h, w), dtype=np.int64)
        
        for color, class_id in self.color_to_class.items():
            # Create mask for this color (BGR format in OpenCV)
            color_mask_bool = (color_mask[:, :, 0] == color[0]) & \
                             (color_mask[:, :, 1] == color[1]) & \
                             (color_mask[:, :, 2] == color[2])
            class_mask[color_mask_bool] = class_id
        
        return class_mask

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        name = self.image_list[idx]
        if name.endswith('.png'):
            name = name[:-4]

        # Construct paths
        img_path = os.path.join(self.image_dir, name + ".png")
        mask_path = os.path.join(self.mask_dir, name + ".png")

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_COLOR)  # Read color mask for multi-class

        # Resize to target size (configurable)
        if self.target_size is not None:
            image = cv2.resize(image, self.target_size, interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, self.target_size, interpolation=cv2.INTER_NEAREST)

        # Normalize image
        image = image.astype(np.float32) / 255.0
        
        # Convert color mask to class indices
        mask = self._color_to_class_mask(mask)

        # Add channel dimension to image
        image = np.expand_dims(image, axis=0)

        if self.transform:
            image, mask = self.transform(image, mask)

        return (torch.tensor(image, dtype=torch.float32), 
                torch.tensor(mask, dtype=torch.long),  # Use long for class indices
                name + ".png")

