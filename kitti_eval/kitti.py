"""
Author: Luigi Piccinelli
Licensed under the CC-BY NC 4.0 license (http://creativecommons.org/licenses/by-nc/4.0/)
"""

import os

import numpy as np
import torch
from PIL import Image

# from .dataset import BaseDataset
from torch.utils.data import Dataset

class KITTIDataset(Dataset):
    
    min_depth = 0.01
    max_depth = 80

    #Original
    test_split = "kitti_eigen_test.txt"
    # train_split = "kitti_eigen_train.txt"

    #Raw Lidar - w/o 360
    # train_split = "kitti_eigen_train_raw.txt"

    #Raw Lidar - w/ 360
    # train_split = "kitti360_eigen_train_raw.txt"

    #Filtered Lidar w/o 360
    train_split = "kitti_eigen_train_filter.txt"

    #Raw+Semi
    # train_split = "kitti360_eigen_train_raw+semi.txt"

    

    def __init__(
        self,
        gt_depth_file,
        base_path,
        depth_scale=256,
        test_depth='raw',
        **kwargs
    ):
        super().__init__()
        self.base_path = base_path
        self.depth_scale = depth_scale
        self.test_depth = test_depth
        self.split_file = gt_depth_file

        # load annotations
        self.dataset = []
        self.load_dataset()
        
        print("Split file used: ", self.split_file)

    def load_dataset(self):
        self.invalid_depth_num = 0
        with open(os.path.join(self.base_path, self.split_file)) as f:
            for line in f:
                img_info = dict()
                # if not self.benchmark:  # benchmark test
                depth_map = line.strip().split(" ")[1]
                if depth_map == "None" or not os.path.exists(
                    os.path.join(self.base_path, depth_map)
                ):
                    self.invalid_depth_num += 1
                    continue
                img_info["annotation_filename_depth"] = os.path.join(
                    self.base_path, depth_map
                )
                if "stereo" in self.split_file:
                    if 'fore' in self.split_file:
                        img_info["in_depth"] = img_info["annotation_filename_depth"].replace('groundtruth_disp_fore', f"groundtruth_{self.test_depth}")
                    elif 'back' in self.split_file:
                        img_info["in_depth"] = img_info["annotation_filename_depth"].replace('groundtruth_disp_back', f"groundtruth_{self.test_depth}")
                    else:
                        img_info["in_depth"] = img_info["annotation_filename_depth"].replace('groundtruth_disp', f"groundtruth_{self.test_depth}")
                else:
                    img_info["in_depth"] = img_info["annotation_filename_depth"].replace('groundtruth', f"groundtruth_{self.test_depth}")
                self.dataset.append(img_info)

        print(
            f"Loaded {len(self.dataset)} images. Totally {self.invalid_depth_num} invalid pairs are filtered"
        )

    def __len__(self):
        """Total number of samples of data."""
        return len(self.dataset)

    def __getitem__(self, idx):
        """Get training/test data after pipeline.
        Args:
            idx (int): Index of data.
        Returns:
            dict: Training/test data (with annotation if `test_mode` is set
                False).
        """
        semidense_depth = (
            np.asarray(
                Image.open(
                    os.path.join(
                        self.base_path,
                        self.dataset[idx]["annotation_filename_depth"],
                    )
                )
            ).astype(np.float32)
            / self.depth_scale
        )

        depth = (
            np.asarray(
                Image.open(
                    os.path.join(
                        self.base_path,
                        self.dataset[idx]["in_depth"],
                    )
                )
            ).astype(np.float32)
            / self.depth_scale
        )
    
        return semidense_depth, depth

   