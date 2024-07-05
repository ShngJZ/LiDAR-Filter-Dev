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
torch.manual_seed(0)
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

        self.inv = kwargs.get("inv", False)

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
                    if self.test_depth == "semi":
                        if 'fore' in self.split_file:
                            img_info["in_depth"] = img_info["annotation_filename_depth"].replace('groundtruth_disp_fore', f"groundtruth")
                        elif 'back' in self.split_file:
                            img_info["in_depth"] = img_info["annotation_filename_depth"].replace('groundtruth_disp_back', f"groundtruth")
                        else:
                            img_info["in_depth"] = img_info["annotation_filename_depth"].replace('groundtruth_disp', f"groundtruth")
                    else:
                        if 'fore' in self.split_file:
                            img_info["in_depth"] = img_info["annotation_filename_depth"].replace('groundtruth_disp_fore', f"groundtruth_{self.test_depth}")
                        elif 'back' in self.split_file:
                            img_info["in_depth"] = img_info["annotation_filename_depth"].replace('groundtruth_disp_back', f"groundtruth_{self.test_depth}")
                        else:
                            img_info["in_depth"] = img_info["annotation_filename_depth"].replace('groundtruth_disp', f"groundtruth_{self.test_depth}")
                else:
                    if self.test_depth == "random" or (self.inv and self.test_depth in ["filter", "half_occ"]):
                        img_info["in_depth_raw"] = img_info["annotation_filename_depth"].replace('groundtruth', f"groundtruth_raw")
                        if self.test_depth == "random":
                            img_info["in_depth_clean"] = img_info["annotation_filename_depth"].replace('groundtruth', f"groundtruth_filter")
                        else:
                            img_info["in_depth_clean"] = img_info["annotation_filename_depth"].replace('groundtruth', f"groundtruth_{self.test_depth}")
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
        # if self.test_depth != "random":
        #     depth = (
        #         np.asarray(
        #             Image.open(
        #                 os.path.join(
        #                     self.base_path,
        #                     self.dataset[idx]["in_depth"],
        #                 )
        #             )
        #         ).astype(np.float32)
        #         / self.depth_scale
        #     )
        if (self.test_depth == "random") or (self.inv and self.test_depth in ["filter", "half_occ"]):
            raw_depth = (
                np.asarray(
                    Image.open(
                        os.path.join(
                            self.base_path,
                            self.dataset[idx]["in_depth_raw"],
                        )
                    )
                ).astype(np.float32)
                / self.depth_scale
            )
            clean_depth = (
                np.asarray(
                    Image.open(
                        os.path.join(
                            self.base_path,
                            self.dataset[idx]["in_depth_clean"],
                        )
                    )
                ).astype(np.float32)
                / self.depth_scale
            )

            if self.test_depth == "random":
                raw_pts = (raw_depth > 0).astype(np.uint8)
                clean_pts = (clean_depth > 0).astype(np.uint8)
                num_pts_raw = np.sum(raw_pts)
                num_pts_clean = np.sum(clean_pts)
                prob_rm = (num_pts_clean/num_pts_raw)
                mask = (raw_pts*np.random.rand(*raw_pts.shape)) < prob_rm
                depth = raw_depth * mask
            else:
                raw_mask = (raw_depth > 0)
                clean_mask = (clean_depth > 0)
                pts_rm_mask = (raw_mask & ~clean_mask)
                depth = raw_depth * pts_rm_mask
        else:
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
            # print(np.sum(depth>0), num_pts_raw, num_pts_clean)
        return semidense_depth, depth

   