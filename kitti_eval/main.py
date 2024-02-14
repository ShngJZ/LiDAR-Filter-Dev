import os
import argparse

from kitti import KITTIDataset

import numpy as np
import torch
from metrics import DICT_METRICS_DEPTH, RunningMetric

from torch.utils.data import DataLoader, SequentialSampler

from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

def main(args):
    min_depth = 0.01
    max_depth = 80

    eval_dataset = KITTIDataset(gt_depth_file=args.semidense_file, 
                           base_path=args.data_path,
                           test_depth=args.test_depth)

    eval_sampler = SequentialSampler(eval_dataset)
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=1,
        num_workers=16,
        sampler=eval_sampler,
        pin_memory=True,
        drop_last=False
    )

    metrics_tracker = RunningMetric(list(DICT_METRICS_DEPTH.keys()))

    for batch in tqdm(eval_loader):
        gt, depth = batch
        mask = (gt > min_depth) & (depth > min_depth)
        # mask = mask & (depth < max_depth) & (gt < max_depth)
        metrics_tracker.accumulate_metrics(gt=gt.to(device), pred=depth.to(device), mask=mask.to(device))
        # print(metrics_tracker.get_metrics()["rmse"])
    print(metrics_tracker.get_metrics())
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Testing", conflict_handler="resolve")

    parser.add_argument("--data-path", default="")
    parser.add_argument("--semidense-file", type=str)
    parser.add_argument("--test-depth", type=str, choices=['raw','filter','random','half_occ'])
    # parser.add_argument("--kitti", type=str, choices=['raw','clean','semidense'], required=False)
    # parser.add_argument("--kitti_stereo", type=str, choices=['foreground','background','all'], required=False)
    # parser.add_argument("--eval_set", type=str, choices=['kitti_stereo','kitti360', 'kitti'], required=False)
    # parser.add_argument("--kitti360", type=str, choices=['raw','clean'], required=False)

    args = parser.parse_args()
    main(args)