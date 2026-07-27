"""
Custom dataloader for MyData dataset

Created for custom Aadhaar card manipulation detection
"""
from project_config import project_root, dataset_paths
from dataset.AbstractDataset import AbstractDataset

import os
import numpy as np
from PIL import Image


class MyData(AbstractDataset):
    """
    MyData directory structure:
    MyData/Data/
    ├── train/
    │   ├── images/
    │   └── mask/
    └── val/
        ├── images/
        └── mask/
    """
    def __init__(self, crop_size, grid_crop, img_list: str, max_dim=None, aug=None):
        """
        Initialize MyData dataset

        Args:
            crop_size: (H, W) or None. H and W must be multiple of 8 if grid_crop==True
            grid_crop: True: crop within 8x8 grid. False: crop anywhere
            img_list: path to text file containing image,mask pairs (one per line)
            max_dim: if image is bigger than this size, it is cropped
            aug: augmentation pipeline
        """
        super().__init__(crop_size, grid_crop, max_dim, aug=aug)
        self._root_path = dataset_paths['MyData']

        # Read image list file
        with open(project_root / img_list, "r") as f:
            self.img_list = [t.strip().split(',') for t in f.readlines()]

        print(f"MyData dataset initialized with {len(self.img_list)} samples")


    def get_img(self, index):
        """
        Load image and mask pair at given index

        Args:
            index: index of the sample to load

        Returns:
            tensor dictionary containing 'img', 'mask', 'index'
        """
        assert 0 <= index < len(self.img_list), f"Index {index} is not available!"

        # Get paths
        rgb_path  = os.path.join(self._root_path, self.img_list[index][0])
        mask_path = os.path.join(self._root_path, self.img_list[index][1])

        # Load mask and convert to binary (0 or 1)
        mask = np.array(Image.open(mask_path).convert("L"))
        mask[mask > 0] = 1  # Binary conversion: any non-zero value becomes 1

        img = Image.open(rgb_path).convert("RGB")
        img_array = np.array(img)
        h, w = img_array.shape[:2]
        if mask.shape != (h, w):
            # print(f"IMAGE PIL DIMENTION: {img.size}")
            # print(f"IMAGE ARRAY DIMENTION: {img_array.shape[:2]}")
            print(f"MASK MISMATCH: Transposing mask from {mask.shape} to match image: ({h}, {w})")
            mask = mask.T
            # print(f"AFTER MASK TRANSPOSE: {mask.shape}")

        assert mask.shape == (h, w), f"MASK MISMATCH: image shape: ({h}, {w}), mask shape {mask.shape}"


        # Verify paths exist
        assert os.path.isfile(rgb_path), f"Image not found: {rgb_path}"
        assert os.path.isfile(mask_path), f"Mask not found: {mask_path}"

        # Create tensor using parent class method
        return self._create_tensor(mask=mask, rgb_path=rgb_path)


    def __len__(self):
        """Return total number of samples"""
        return len(self.img_list)
