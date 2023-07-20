import os, time, tqdm, glob, natsort, copy
import numpy as np
from PIL import Image
from core.lidar_cleaner import LiDARCleaner
prj_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

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

BASE_PATH, DIR, SUBDIR, cam_channel = os.path.join(prj_root, "misc/Kitti"), "2011_09_26", "2011_09_26_drive_0052_sync", 'image_02'

LIDAR_DIR = os.path.join(BASE_PATH, DIR, SUBDIR, 'velodyne_points/data')
IMG_DIR = os.path.join(BASE_PATH, DIR, SUBDIR, cam_channel, 'data')
frame_list = glob.glob(os.path.join(IMG_DIR, '*.png'))
frame_list = natsort.natsorted(frame_list)
fr_num = len(frame_list)
dr, cnt = 0, 0

EXPORT_ROOT = os.path.join(prj_root, "misc/midpred", "occvls_kitti")
os.makedirs(EXPORT_ROOT, exist_ok=True)

# Pad LiDAR Scan To Logical Coordinates Center
x, y, z = 0.28, -0.00, -0.17
T_LiDAR_Padding = geometric_transformation(np.eye(3), np.array([x, y, z]))

for ii, frame_path in enumerate(tqdm.tqdm(frame_list)):
    frame_name = os.path.basename(frame_path)
    fr_num = frame_name.split('.')[0]
    LIDAR_PATH, LIDAR_CALIB_PATH, CAM_CALIB_PATH = \
        os.path.join(LIDAR_DIR, f'{fr_num}.bin'), os.path.join(BASE_PATH, DIR, 'calib_velo_to_cam.txt'), os.path.join(BASE_PATH, DIR, 'calib_cam_to_cam.txt')

    im, lidar_pts = Image.open(frame_path), read_LiDAR(LIDAR_PATH)
    lidar_pts = read_LiDAR(LIDAR_PATH)
    lidar_to_cam_ref = read_LiDAR_calib(LIDAR_CALIB_PATH)
    cam_intrinsic, cam_ref_to_cam = read_CAM_calib(CAM_CALIB_PATH)

    extrinsic_LiDAR2Cam = cam_ref_to_cam @ lidar_to_cam_ref
    intrinsic_LiDAR2Cam = cam_intrinsic

    # Apply Padding
    extrinsic_LiDAR2Cam, lidar_pts = extrinsic_LiDAR2Cam @ np.linalg.inv(T_LiDAR_Padding), (T_LiDAR_Padding @ lidar_pts.T).T

    # Apply Filtering
    w, h = im.size
    cleaner = LiDARCleaner(
        intrinsic_cam=intrinsic_LiDAR2Cam,
        extrinsic_LiDAR2Cam=extrinsic_LiDAR2Cam[0:3, :],
        LiDARPoints3D=lidar_pts[:, 0:3].T,
        height=h, width=w,
        rszh=1.0, rszw=1.0,
        plotmarker_size=4.0
    )
    visible_points_filtered, lidar_to_be_occluded, imcombined = cleaner(rgb=im, debug=True)
    imcombined.save(os.path.join(EXPORT_ROOT, frame_name))

print("Generated %d Samples, Ave Run time %f sec" % (cnt, dr / cnt))