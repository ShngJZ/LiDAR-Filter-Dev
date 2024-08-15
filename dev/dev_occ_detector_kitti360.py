import os, time, glob, natsort, copy
from tqdm import tqdm
import numpy as np
from PIL import Image
import torch
import sys
sys.path.append("/user/ganesang/cvl/LidarFilter/LiDAR-Filter-Dev/")
from core.lidar_cleaner import LiDARCleaner
from viewer import projectVeloToImage, Kitti360Viewer3DRaw
from loadCalibration import loadCalibrationCameraToPose, loadCalibrationRigid
from camera_helper import CameraPerspective
import h5py
from p_tqdm import p_map

# BASE_DIR = "/scratch1/ganesang/kitti360/"
BASE_DIR = "/scratch0/ganesang/kitti360"
CALIB_DIR = os.path.join(BASE_DIR, "calibration")
DATA_3D = os.path.join(BASE_DIR, 'data_3d_raw')
DATA_2D = os.path.join(BASE_DIR, 'data_2d_raw')

occ_file = h5py.File(os.path.join(DATA_3D, 'occluded_pts_new.h5'), 'w')

def geometric_transformation(rotation, translation):
    mat = np.eye(4)
    mat[:3, :3] = rotation
    mat[:3, 3] = translation
    return mat

def func(DIR):
    print("Sequence: ", DIR)

    kitti360Path = BASE_DIR
    
    seq = int(DIR.split('_')[-2])
    sequence = '2013_05_28_drive_%04d_sync'%seq

    # perspective camera
    cam_id = 0
    camera = CameraPerspective(kitti360Path, sequence, cam_id)

    # object for parsing 3d raw data 
    velo = Kitti360Viewer3DRaw(mode='velodyne', seq=seq, BASE_DIR=BASE_DIR)
    
    # cam_0 to velo
    fileCameraToVelo = os.path.join(kitti360Path, 'calibration', 'calib_cam_to_velo.txt')
    TrCam0ToVelo = loadCalibrationRigid(fileCameraToVelo)

    # all cameras to system center 
    fileCameraToPose = os.path.join(kitti360Path, 'calibration', 'calib_cam_to_pose.txt')
    TrCamToPose = loadCalibrationCameraToPose(fileCameraToPose)

    # velodyne to all cameras
    TrVeloToCam = {}
    for k, v in TrCamToPose.items():
        # Tr(cam_k -> velo) = Tr(cam_k -> cam_0) @ Tr(cam_0 -> velo)
        TrCamkToCam0 = np.linalg.inv(TrCamToPose['image_00']) @ TrCamToPose[k]
        TrCamToVelo = TrCam0ToVelo @ TrCamkToCam0
        # Tr(velo -> cam_k)
        TrVeloToCam[k] = np.linalg.inv(TrCamToVelo)
    
    # take the rectification into account for perspective cameras
    if cam_id==0 or cam_id == 1:
        TrVeloToRect = np.matmul(camera.R_rect, TrVeloToCam['image_%02d' % cam_id])
    else:
        TrVeloToRect = TrVeloToCam['image_%02d' % cam_id]

    # color map for visualizing depth map
    # cm = plt.get_cmap('jet')

    # visualize a set of frame
    # for each frame, load the raw 3D scan and project to image plane
    
    LIDAR_DIR = os.path.join(DATA_3D, DIR, "velodyne_points/data")
    frame_list = [f.split(".")[0] for f in os.listdir(LIDAR_DIR)]
    OUTPUT_DIR = os.path.join(DATA_3D, DIR, "occ_pts")
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
    except OSError as error:
        print(error)

    for frame in tqdm(frame_list):
        out_file_path = os.path.join(OUTPUT_DIR, f"{frame}.npy")
        if os.path.exists(out_file_path):
            continue
        points = velo.loadVelodyneData(int(frame))
        points[:,3] = 1

        # Translation adjustment
        x,y,z = 0.24, 0.17, -0.21
        T = geometric_transformation(np.eye(3), np.array([x, y, z]))
        inv_T = np.linalg.inv(T)

        extrinsic_LiDAR2Cam = TrVeloToRect.copy()
        cam_intrinsic = camera.K.copy()

        cleaner = LiDARCleaner(
            intrinsic_cam=cam_intrinsic[:3,:3],
            extrinsic_LiDAR2Cam=(extrinsic_LiDAR2Cam@inv_T)[0:3, :],
            LiDARPoints3D=(T@points.T)[0:3, :],
            height=camera.height, width=camera.width,
            rszh=1, rszw=1
        )
        visible_points_filtered, lidar_to_be_occluded = cleaner(debug=False)

        #TO SAVE .npy
        # np.save(out_file_path, lidar_to_be_occluded.cpu().numpy())

        #TO SAVE .h5
        # data = np.load(out_file_path)
        # occ_file[f"{DIR}@{frame}"] = data
        
    return 0

def main():
    dir_list = [file for file in os.listdir(DATA_3D) if os.path.isdir(os.path.join(DATA_3D, file))]
    print(dir_list)

    #TO SAVE .npy
    # r = p_map(func, dir_list, num_cpus=1)#os.cpu_count())

    #TO SAVE .h5
    # for dir_name in dir_list:
    #     func(dir_name)
    # occ_file.close()


if __name__ == "__main__":
    main()