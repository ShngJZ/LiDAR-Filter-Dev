import torch
from metrics import DICT_METRICS_DEPTH, RunningMetric
from PIL import Image
import numpy as np
import os

# torch.manual_seed(0)
# ex_gt = torch.rand(1,130,130)*80
# ex_pred = ex_gt + torch.rand(1,130,130)

# ex_mask = torch.randint(0,2,(1,5,5))

# ex_gt_back = ex_gt.clone()
# ex_gt_back[0,25:40,25:40] = 0
# ex_gt_fore = ex_gt - ex_gt_back

# metrics_tracker = RunningMetric(list(DICT_METRICS_DEPTH.keys()))
# metrics_tracker_fore = RunningMetric(list(DICT_METRICS_DEPTH.keys()))
# metrics_tracker_back = RunningMetric(list(DICT_METRICS_DEPTH.keys()))
# for i in range(ex_gt.shape[0]):
#     gt = ex_gt[[i]]
#     pred = ex_pred[[i]]
#     mask = (gt > 0) & (pred > 0)
#     metrics_tracker.accumulate_metrics(gt=gt, pred=pred, mask=mask)
# print(metrics_tracker.get_metrics()['rmse'])

# for i in range(ex_gt.shape[0]):
#     gt = ex_gt_fore[[i]]
#     pred = ex_pred[[i]]
#     mask = (gt > 0) & (pred > 0)
#     metrics_tracker_fore.accumulate_metrics(gt=gt, pred=pred, mask=mask)
# print(metrics_tracker_fore.get_metrics()['rmse'])

# for i in range(ex_gt.shape[0]):
#     gt = ex_gt_back[[i]]
#     pred = ex_pred[[i]]
#     mask = (gt > 0) & (pred > 0)
#     metrics_tracker_back.accumulate_metrics(gt=gt, pred=pred, mask=mask)
# print(metrics_tracker_back.get_metrics()['rmse'])

def load_depth_img(path):
    img = Image.open(path)
    img_arr = np.array(img).astype(np.float32)/256
    return torch.tensor(img_arr[None, :])

KITTI_DIR = "/scratch1/ganesang/kitti/datasets"
split_file = "splits/kitti_stereo_test.txt"
with open(split_file, "r") as f:
    data = [line.strip().split(" ") for line in f]

idx = 5
depth_file = data[idx][1]
depth_path = os.path.join(KITTI_DIR, depth_file)
depth_fore_path = depth_path.replace("groundtruth_disp", "groundtruth_disp_fore")
depth_back_path = depth_path.replace("groundtruth_disp", "groundtruth_disp_back")
depth_raw_path = depth_path.replace("groundtruth_disp", "groundtruth_raw")
depth_clean_path = depth_path.replace("groundtruth_disp", "groundtruth_filter")

gt_disp = load_depth_img(depth_path)
gt_disp_fore = load_depth_img(depth_fore_path)
gt_disp_back = load_depth_img(depth_back_path)
depth_raw = load_depth_img(depth_raw_path)
depth_clean = load_depth_img(depth_clean_path)

mask_full = (gt_disp > 0) #& (depth_raw > 0)
mask_fore = (gt_disp_fore > 0) #& (depth_raw > 0)
mask_back = (gt_disp_back > 0) #& (depth_raw > 0)

num_full = torch.sum(mask_full)
num_fore = torch.sum(mask_fore)
num_back = torch.sum(mask_back)

print(num_full)
print(num_fore)
print(num_back)

metrics_rmse_full = RunningMetric(list(["rmse", "d1"]))
metrics_rmse_fore = RunningMetric(list(["rmse"]))
metrics_rmse_back = RunningMetric(list(["rmse"]))

print(gt_disp.shape, depth_raw.shape, mask_full.shape)
metrics_rmse_full.accumulate_metrics(gt=gt_disp, pred=depth_raw, mask=mask_full)
metrics_rmse_fore.accumulate_metrics(gt=gt_disp_fore, pred=depth_raw, mask=mask_fore)
metrics_rmse_back.accumulate_metrics(gt=gt_disp_back, pred=depth_raw, mask=mask_back)

rmse_full = metrics_rmse_full.get_metrics()["rmse"]
rmse_fore = metrics_rmse_fore.get_metrics()["rmse"]
rmse_back = metrics_rmse_back.get_metrics()["rmse"]
print("Full: ", rmse_full)
print("Fore: ", rmse_fore)
print("Back: ", rmse_back)

print(num_fore*(rmse_fore**2) + num_back*(rmse_back))
print(num_full*(rmse_full**2))