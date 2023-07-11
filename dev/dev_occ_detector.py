import os, time
import torch
import numpy as np
import tqdm
from PIL import Image
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion
# import sys
# sys.path.insert(0,'/user/ganesang/cvl/LidarFilter/LiDAR-Filter-Dev/')
from core.lidar_cleaner import LiDARCleaner

def rotation_matrix(theta_x, theta_y, theta_z):
    theta_x = (np.pi / 180) * theta_x
    theta_y = (np.pi / 180) * theta_y
    theta_z = (np.pi / 180) * theta_z
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(theta_x), -np.sin(theta_x)],
                   [0, np.sin(theta_x), np.cos(theta_x)]])
    Ry = np.array([[np.cos(theta_y), 0, np.sin(theta_y)],
                   [0, 1, 0],
                   [-np.sin(theta_y), 0, np.cos(theta_y)]])
    Rz = np.array([[np.cos(theta_z), -np.sin(theta_z), 0],
                   [np.sin(theta_z), np.cos(theta_z), 0],
                   [0, 0, 1]])

    R = Rx @ Ry @ Rz

    return R

def geometric_transformation(rotation, translation):
    mat = np.eye(4)
    mat[:3, :3] = rotation
    mat[:3, 3] = translation
    return mat

def read_intrinsic_extrinsic_LiDAR2Cam(nusc, lidar_meta, cam_meta):
    cs_record_lidar = nusc.get('calibrated_sensor', lidar_meta['calibrated_sensor_token'])
    cs_record_cam = nusc.get('calibrated_sensor', cam_meta['calibrated_sensor_token'])
    poserecord_lidar = nusc.get('ego_pose', lidar_meta['ego_pose_token'])
    poserecord_cam = nusc.get('ego_pose', cam_meta['ego_pose_token'])

    lidar_to_veh_rot = Quaternion(cs_record_lidar['rotation']).rotation_matrix
    lidar_to_veh_trans = np.array(cs_record_lidar['translation'])
    lidar_to_veh_mat = geometric_transformation(lidar_to_veh_rot, lidar_to_veh_trans)

    cam_to_veh_rot = Quaternion(cs_record_cam['rotation']).rotation_matrix
    cam_to_veh_trans = np.array(cs_record_cam['translation'])
    cam_to_veh_mat = geometric_transformation(cam_to_veh_rot, cam_to_veh_trans)

    veh_to_glb_lidar_rot = Quaternion(poserecord_lidar['rotation']).rotation_matrix
    veh_to_glb_lidar_trans = np.array(poserecord_lidar['translation'])
    veh_to_glb_lidar_mat = geometric_transformation(veh_to_glb_lidar_rot, veh_to_glb_lidar_trans)

    veh_to_glb_cam_rot = Quaternion(poserecord_cam['rotation']).rotation_matrix
    veh_to_glb_cam_trans = np.array(poserecord_cam['translation'])
    veh_to_glb_cam_mat = geometric_transformation(veh_to_glb_cam_rot, veh_to_glb_cam_trans)

    # Relative pose from lidar to rgb camera
    extrinsic_LiDAR2Cam = np.linalg.inv(cam_to_veh_mat) @ np.linalg.inv(veh_to_glb_cam_mat) @ veh_to_glb_lidar_mat @ lidar_to_veh_mat
    intrinsic = np.array(cs_record_cam['camera_intrinsic'])

    return intrinsic, extrinsic_LiDAR2Cam

nusc = NuScenes(version='v1.0-trainval', dataroot='/scratch1/ganesang/nuScenes/', verbose=True)

dr, cnt = 0, 0

for ii in tqdm.tqdm(range(len(nusc.sample))):
    for cam_channel in ['CAM_FRONT', 'CAM_BACK', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']:
        export_root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'nuScenes', 'filtered_pts')
        os.makedirs(export_root, exist_ok=True)

        sample = nusc.sample[ii]

        token = sample['token']

        lidar_token = sample['data']['LIDAR_TOP']
        cam_token = sample['data'][cam_channel]

        lidar_meta = nusc.get('sample_data', lidar_token)
        cam_meta = nusc.get('sample_data', cam_token)
        pc = LidarPointCloud.from_file(os.path.join(nusc.dataroot, lidar_meta['filename']))
        im = Image.open(os.path.join(nusc.dataroot, cam_meta['filename']))

        w, h = im.size
        intrinsic, extrinsic_LiDAR2Cam = read_intrinsic_extrinsic_LiDAR2Cam(nusc, lidar_meta, cam_meta)

        cleaner = LiDARCleaner(
            intrinsic_cam=intrinsic,
            extrinsic_LiDAR2Cam=extrinsic_LiDAR2Cam[0:3, :],
            LiDARPoints3D=pc.points[0:3, :],
            height=h, width=w,
            rszh=0.2, rszw=0.5
        )

        if np.mod(ii, 1) == 50:
            visible_points_filtered, imcombined = cleaner(rgb=im, debug=True)
            imcombined.save(os.path.join(export_root, '{}_{}.jpg'.format(cam_channel, str(ii))))
            cnt += 1
        else:
            st = time.time()
            visible_points_filtered, camprj_vls, camdepth_vls = cleaner(rgb=im, debug=False)
            final_pts = torch.concatenate([camprj_vls.T, camdepth_vls[:,None]], dim=1)
            final_pts = final_pts[visible_points_filtered]
            lidar_file = lidar_meta['filename'].split('/')[-1]
            cam_file = cam_meta['filename'].split('/')[-1]
            torch.save(data, os.path.join(export_root, f"{lidar_file}#{cam_file}.pt"))
            dr += time.time() - st
            cnt += 1
        


print("Generated %d Samples, Ave Run time %f sec" % (cnt, dr / cnt))