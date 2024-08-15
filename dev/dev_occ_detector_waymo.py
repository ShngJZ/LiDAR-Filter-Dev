import os, time, tqdm, glob, copy#, natsort
import numpy as np
from PIL import Image
import sys
sys.path.append("/user/ganesang/cvl/LidarFilter/LiDAR-Filter-Dev/")
from core.lidar_cleaner import LiDARCleaner
prj_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
import h5py
import shutil
from p_tqdm import p_map, p_umap, p_imap
from multiprocessing import Queue, Pool, current_process

NUM_GPUS = 8
PROC_PER_GPU = 6

queue = Queue()

def read_LiDAR(LIDAR_PATH):
    lidar_pts = np.fromfile(LIDAR_PATH,dtype=np.float32).reshape((-1, 4))[:,:3]
    lidar_pts = np.concatenate([lidar_pts, np.ones((len(lidar_pts), 1))], axis=1)
    return lidar_pts

def geometric_transformation(rotation, translation):
    mat = np.eye(4)
    mat[:3, :3] = rotation
    mat[:3, 3] = translation
    return mat

def get_calib_data(CALIB_PATH):
    calib_dict = {}
    with open(CALIB_PATH, 'r') as f:
        for line in f:
            if line == "\n":
                continue
            key, val = line.strip().split(": ")
            if "P" in key:
                calib_dict[key] = np.array(val.split()).astype(np.float32).reshape(-1,4)
            elif "Tr" in key:
                mat = np.array(val.split()).astype(np.float32).reshape(-1,4)
                calib_dict[key] = geometric_transformation(mat[:3,:3],mat[:3,-1])
            elif "R" in key:
                calib_dict[key] = geometric_transformation(np.array(val.split()).astype(np.float32).reshape(-1,3),0)
            else:
                calib_dict[key] = val

    return calib_dict

DATA_DIR = "/scratch0/ganesang/waymo/waymo_open_dataset_kitti/"
# DATA_DIR = "/user/ganesang/cvl/mmdetection3d/data/waymo/kitti_format"
SPLIT_DIR = "training"#"validation"
IMG_DIR = os.path.join(DATA_DIR, SPLIT_DIR, "image_0")
CALIB_DIR = os.path.join(DATA_DIR, SPLIT_DIR, "calib")
LIDAR_DIR = os.path.join(DATA_DIR, SPLIT_DIR, "velodyne")
LABEL_DIR = os.path.join(DATA_DIR, SPLIT_DIR, "label_0")

frame_list = [fr.split('.')[0] for fr in os.listdir(LABEL_DIR)]
frame_list.sort()
# occ_file = h5py.File(os.path.join(DATA_DIR, SPLIT_DIR,'occ_pts.h5'), 'w')

OUT_DIR = "/scratch0/ganesang/waymo/waymo_open_dataset_kitti"
occ_file = h5py.File(os.path.join(OUT_DIR, 'occ_pts.h5'), 'a')

dr, cnt = 0, 0

x, y, z = -1.45, -0.03, -2.20
T_LiDAR_Padding = geometric_transformation(np.eye(3), np.array([x, y, z]))

def func(frame_num):
    out_file_path = os.path.join(DATA_DIR, SPLIT_DIR, "occ_pts",f"{frame_num}.npy")

    # TO SAVE .npy

    if os.path.exists(out_file_path):
        return
    IMG_PATH = os.path.join(IMG_DIR, f'{frame_num}.png')
    # IMG_PATH = os.path.join(IMG_DIR, f'{frame_num}.jpg')
    LIDAR_PATH = os.path.join(LIDAR_DIR, f'{frame_num}.bin')
    CALIB_PATH = os.path.join(CALIB_DIR, f'{frame_num}.txt')

    im, lidar_pts = Image.open(IMG_PATH), read_LiDAR(LIDAR_PATH)
    calib_dict = get_calib_data(CALIB_PATH)
    
    Tr_velo2cam = calib_dict["Tr_velo_to_cam"]
    # Tr_velo2cam = calib_dict["Tr_velo_to_cam_0"]
    R0_rect = calib_dict["R0_rect"]
    P2_mat = calib_dict["P2"]
    intrinsic_LiDAR2Cam = P2_mat.copy()[:3,:3]
    extrinsic_LiDAR2Cam = (R0_rect @ Tr_velo2cam).copy()

    # Apply Padding
    extrinsic_LiDAR2Cam, lidar_pts = extrinsic_LiDAR2Cam @ np.linalg.inv(T_LiDAR_Padding), (T_LiDAR_Padding @ lidar_pts.T).T

    # Apply Filtering
    w, h = im.size
    gpu_id = int(frame_num) % 8
    cleaner = LiDARCleaner(
        intrinsic_cam=intrinsic_LiDAR2Cam,
        extrinsic_LiDAR2Cam=extrinsic_LiDAR2Cam[0:3, :],
        LiDARPoints3D=lidar_pts[:, 0:3].T,
        height=h, width=w,
        rszh=1.0, rszw=1.0,
        plotmarker_size=1.0, gpu_id=gpu_id
    )

    _, lidar_to_be_occluded = cleaner(debug=False)
    np.save(out_file_path, lidar_to_be_occluded.numpy().astype(bool))



    # TO SAVE .h5 from .npy
    # data = np.load(out_file_path)
    # occ_file[f"{frame_num}"] = data

if __name__ == "__main__":
    # TO SAVE .npy
    r = p_map(func, frame_list, num_cpus=8)
    
    # TO SAVE .h5 from .npy
    # for fr in tqdm.tqdm(frame_list):
    #     func(fr)
    # occ_file.close()
