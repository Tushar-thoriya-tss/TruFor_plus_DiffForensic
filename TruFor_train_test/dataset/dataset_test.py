# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Copyright (c) 2023 Image Processing Research Group of University Federico II of Naples ('GRIP-UNINA').
#
# All rights reserved.
# This work should only be used for nonprofit purposes.
#
# By downloading and/or using any of these files, you implicitly agree to all the
# terms of the license, as specified in the document LICENSE.txt
# (included in this package) and online at
# http://www.grip.unina.it/download/LICENSE_OPEN.txt

"""
Created in September 2022
@author: fabrizio.guillaro
"""

from torch.utils.data import Dataset
import random
import numpy as np
import torch
from PIL import Image


class TestDataset(Dataset):
    def __init__(self, list_img=None):
        self.img_list = list_img

    def shuffle(self):
        random.shuffle(self.img_list)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        assert self.img_list
        assert 0 <= index < len(self.img_list), f"Index {index} is not available!"
        rgb_path = self.img_list[index]
        img_RGB = np.array(Image.open(rgb_path).convert("RGB"))
        return torch.tensor(img_RGB.transpose(2, 0, 1), dtype=torch.float) / 256.0, rgb_path

    def get_filename(self, index):
        item = self.img_list[index]
        if isinstance(item, list):
            return item[0]
        else:
            return item



# class TestDataset(Dataset):
#     def __init__(self, list_img=None, max_side=768):
#         self.img_list = list_img
#         self.max_side = max_side  # resize only if max(h, w) > max_side

#     def shuffle(self):
#         random.shuffle(self.img_list)

#     def __len__(self):
#         return len(self.img_list)

#     def __getitem__(self, index):
#         assert self.img_list
#         assert 0 <= index < len(self.img_list), f"Index {index} is not available!"
#         rgb_path = self.img_list[index]

#         img = Image.open(rgb_path).convert("RGB")

#         # Preserve original logic (commented):
#         # img_RGB = np.array(Image.open(rgb_path).convert("RGB"))
#         # return torch.tensor(img_RGB.transpose(2, 0, 1), dtype=torch.float) / 256.0, rgb_path

#         w, h = img.size
#         max_dim = max(w, h)
#         if self.max_side is not None and max_dim > self.max_side:
#             scale = self.max_side / float(max_dim)
#             new_w = int(round(w * scale))
#             new_h = int(round(h * scale))
#             img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

#         img_RGB = np.array(img)
#         return torch.tensor(img_RGB.transpose(2, 0, 1), dtype=torch.float) / 256.0, rgb_path

#     def get_filename(self, index):
#         item = self.img_list[index]
#         if isinstance(item, list):
#             return item[0]
#         else:
#             return item
