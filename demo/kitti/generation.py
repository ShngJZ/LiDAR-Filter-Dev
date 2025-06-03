import os
import os.path as osp
import numpy as np
from PIL import Image
from tqdm import tqdm
import sys

# Add custom module path
sys.path.append("../../LiDAR-Filter-Dev")
from core.lidar_cleaner import LiDARCleaner
import argparse

def geometric_transformation(rotation, translation):
    """Create 4x4 transformation matrix from rotation and translation"""
    mat = np.eye(4)
    mat[:3, :3] = rotation  # Set rotation part
    mat[:3, 3] = translation  # Set translation part
    return mat

def read_CAM_calib(CAM_CALIB_PATH):
    """Read camera calibration file and return intrinsic and reference transformations"""
    cam_calib = {}
    with open(CAM_CALIB_PATH, 'r') as f:
        for line in f:
            key, value = line.split(': ')
            cam_calib[key] = np.array(value.strip().split())

    # Get camera intrinsic parameters
    cam_intrinsic = cam_calib["P_rect_02"].astype(np.float32).reshape(-1, 4)
    # Get camera reference to camera transformation
    cam_ref_to_cam = geometric_transformation(
        cam_calib["R_rect_00"].astype(np.float32).reshape(-1, 3),
        0
    )

    # Normalize intrinsic matrix
    cam_intrinsic_ = np.eye(3)
    cam_intrinsic_[0:3, 0:3] = cam_intrinsic[0:3, 0:3]
    # Calculate final transformation
    cam_ref_to_cam_ = np.linalg.inv(cam_intrinsic_) @ cam_intrinsic @ cam_ref_to_cam
    return cam_intrinsic_, cam_ref_to_cam_

def read_LiDAR_calib(LIDAR_CALIB_PATH):
    """Read LiDAR calibration file and return transformation matrix"""
    lidar_calib = {}
    with open(LIDAR_CALIB_PATH, 'r') as f:
        for line in f:
            key, value = line.split(': ')
            lidar_calib[key] = np.array(value.strip().split())

    # Get rotation and translation parameters
    lidar_R = lidar_calib['R'].astype(np.float32).reshape(-1, 3)
    lidar_T = lidar_calib['T'].astype(np.float32)
    return geometric_transformation(lidar_R, lidar_T)

def process_frame(fr_num):
    """Process single frame: load data, apply transformations and save result"""
    # Load image and LiDAR data
    im = Image.open(os.path.join(IMG_DIR, f'{fr_num}.png'))
    data_bin = np.fromfile(
        os.path.join(LIDAR_DIR, f'{fr_num}.bin'),
        dtype=np.float32
    ).reshape((-1, 4))

    lidar_pts = data_bin
    lidar_pts[:,-1] = 1.0  # Set homogeneous coordinate
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Pre-calibrated transformation to virtual camera
    x, y, z = 0.28, -0.00, -0.17
    T_LiDAR_Padding = geometric_transformation(np.eye(3), np.array([x, y, z]))

    # Read calibration data
    lidar_to_cam_ref = read_LiDAR_calib(LIDAR_CALIB_PATH)
    cam_intrinsic, cam_ref_to_cam = read_CAM_calib(CAM_CALIB_PATH)

    # Calculate extrinsic transformation
    extrinsic_LiDAR2Cam = cam_ref_to_cam @ lidar_to_cam_ref
    intrinsic_LiDAR2Cam = cam_intrinsic

    # Apply padding transformation
    extrinsic_LiDAR2Cam = extrinsic_LiDAR2Cam @ np.linalg.inv(T_LiDAR_Padding)
    lidar_pts = (T_LiDAR_Padding @ lidar_pts.T).T

    # Process LiDAR points
    w, h = im.size
    cleaner = LiDARCleaner(
        intrinsic_cam=intrinsic_LiDAR2Cam,
        extrinsic_LiDAR2Cam=extrinsic_LiDAR2Cam[0:3, :],
        LiDARPoints3D=lidar_pts[:, 0:3].T,
        height=h, width=w,
        rszh=1.0, rszw=1.0,
        plotmarker_size=2,
        showimage=True
    )

    # Filter points and save result
    visible_points_filtered, lidar_to_be_occluded, imcombined = cleaner(rgb=im, debug=True)
    imcombined.save(osp.join(OUTPUT_DIR, f"{fr_num}.png"))

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Testing", conflict_handler="resolve")
    parser.add_argument("--data_dir", help="Path to KITTI dataset", default='/media/disk5/KITTI')
    parser.add_argument("--seq", default="2011_09_26_drive_0001_sync")
    parser.add_argument("--output_dir", default="/media/disk5/LiDAR-Filter-Dev/demooutput/KITTI_Occlusion")
    args = parser.parse_args()

    # Set up paths
    DATA_DIR = args.data_dir
    SEQ = args.seq
    DIR = SEQ.split("_drive")[0]
    LIDAR_DIR = os.path.join(DATA_DIR, DIR, SEQ, 'velodyne_points/data')
    IMG_DIR = os.path.join(DATA_DIR, DIR, SEQ, 'image_02/data')
    CAM_CALIB_PATH = os.path.join(DATA_DIR, DIR, 'calib_cam_to_cam.txt')
    LIDAR_CALIB_PATH = os.path.join(DATA_DIR, DIR, 'calib_velo_to_cam.txt')
    OUTPUT_DIR = osp.join(args.output_dir, SEQ)

    # Process all frames
    frame_list = [f.split('.')[0] for f in os.listdir(LIDAR_DIR)]
    for fr_num in tqdm(sorted(frame_list)):
        process_frame(fr_num)