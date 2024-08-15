import os
import os.path as osp
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import sys

sys.path.append("../LiDAR-Filter-Dev")
from core.lidar_cleaner import LiDARCleaner
# import argparse
from p_tqdm import p_map
import h5py

def read_LiDAR(LIDAR_PATH):
    lidar_pts = np.fromfile(LIDAR_PATH, dtype=np.float32).reshape((-1, 4))
    lidar_pts[:, 3] = 1.0
    return lidar_pts
def geometric_transformation(rotation, translation):
    mat = np.eye(4)
    mat[:3, :3] = rotation
    mat[:3, 3] = translation
    return mat
def read_LiDAR_calib(LIDAR_CALIB_PATH):
    lidar_calib = {}
    with open(LIDAR_CALIB_PATH, 'r') as f:
        for line in f:
            lidar_calib[line.split(': ')[0]] = np.array(line.split(': ')[1].strip().split())
    lidar_R = lidar_calib['R'].astype(np.float32).reshape(-1, 3)
    lidar_T = lidar_calib['T'].astype(np.float32)
    return geometric_transformation(lidar_R, lidar_T)

def read_CAM_calib(CAM_CALIB_PATH):
    cam_calib = {}
    with open(CAM_CALIB_PATH, 'r') as f:
        for line in f:
            cam_calib[line.split(': ')[0]] = np.array(line.split(': ')[1].strip().split())

    cam_intrinsic = cam_calib["P_rect_02"].astype(np.float32).reshape(-1, 4)
    cam_ref_to_cam = geometric_transformation(cam_calib["R_rect_00"].astype(np.float32).reshape(-1, 3), 0)

    cam_intrinsic_ = np.eye(3)
    cam_intrinsic_[0:3, 0:3] = cam_intrinsic[0:3, 0:3]
    cam_ref_to_cam_ = np.linalg.inv(cam_intrinsic_) @ cam_intrinsic @ cam_ref_to_cam
    return cam_intrinsic_, cam_ref_to_cam_


BASE_DIR = "/scratch0/ganesang/kitti/"
occ_file = h5py.File(os.path.join(BASE_DIR, 'occ_pts.h5'), 'w')

def func(seq_tuple):
    SEQ, gpu_id = seq_tuple
    DIR = SEQ.split("_drive")[0]
    LIDAR_DIR = osp.join(BASE_DIR, DIR, SEQ, 'velodyne_points/data')
    IMG_DIR = osp.join(BASE_DIR, DIR, SEQ, 'image_02/data')
    CAM_CALIB_PATH = osp.join(BASE_DIR, DIR, 'calib_cam_to_cam.txt')
    LIDAR_CALIB_PATH = osp.join(BASE_DIR, DIR, 'calib_velo_to_cam.txt')
    OUTPUT_DIR = osp.join(BASE_DIR, DIR, SEQ, 'occ_pts')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    frame_list = [f.split('.')[0] for f in os.listdir(LIDAR_DIR)]

    for fr_num in frame_list:
        out_file_path = osp.join(OUTPUT_DIR, f"{fr_num}.npy")
        # if osp.exists(out_file_path):
        #     continue
        # im = Image.open(osp.join(IMG_DIR, f'{fr_num}.png'))
        # data_bin = np.fromfile(osp.join(LIDAR_DIR, f'{fr_num}.bin'),dtype=np.float32).reshape((-1, 4))
        # lidar_pts = data_bin
        # lidar_pts[:,-1] = 1.0
        

        # x, y, z = 0.28, -0.00, -0.17
        # T_LiDAR_Padding = geometric_transformation(np.eye(3), np.array([x, y, z]))
        # lidar_to_cam_ref = read_LiDAR_calib(LIDAR_CALIB_PATH)
        # cam_intrinsic, cam_ref_to_cam = read_CAM_calib(CAM_CALIB_PATH)
        # extrinsic_LiDAR2Cam = cam_ref_to_cam @ lidar_to_cam_ref

        # intrinsic_LiDAR2Cam = cam_intrinsic
        # extrinsic_LiDAR2Cam, lidar_pts = extrinsic_LiDAR2Cam @ np.linalg.inv(T_LiDAR_Padding), (T_LiDAR_Padding @ lidar_pts.T).T

        # w, h = im.size
        # cleaner = LiDARCleaner(
        #     intrinsic_cam=intrinsic_LiDAR2Cam,
        #     extrinsic_LiDAR2Cam=extrinsic_LiDAR2Cam[0:3, :],
        #     LiDARPoints3D=lidar_pts[:, 0:3].T,
        #     height=h, width=w,
        #     rszh=1.0, rszw=1.0,
        #     plotmarker_size=2, gpu_id=gpu_id
        # )

        # # visible_points_filtered, lidar_to_be_occluded, imcombined = cleaner(rgb=im, debug=True)
        # visible_points_filtered, lidar_to_be_occluded = cleaner(debug=False)
        # out_file_path = osp.join(OUTPUT_DIR, f"{fr_num}.npy")
        # np.save(out_file_path, lidar_to_be_occluded.numpy().astype(bool))
        data = np.load(out_file_path)
        occ_file[f"{SEQ}@{fr_num}"] = data

    return

dir_list = [d for d in os.listdir(BASE_DIR) if "20" in d]
seq_list = []
cnt = 0
for d in dir_list:
    for s in os.listdir(osp.join(BASE_DIR,d)):
        if os.path.isdir(osp.join(BASE_DIR,d,s)):
            seq_list.append((s,cnt))
            cnt = (cnt+1)%8
            
# r = p_map(func, seq_list, num_cpus=16)
for seq_tuple in tqdm(seq_list):
    func(seq_tuple)

occ_file.close()