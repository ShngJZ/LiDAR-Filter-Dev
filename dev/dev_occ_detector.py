import copy
import os
import torch
import numpy as np
from PIL import Image
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion
import matplotlib.pyplot as plt
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay
from core.occlusion_detector.occlusion_detector import occlusion_detector
from core.utils.visualization import tensor2disp

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


export_root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'misc', 'midpred', 'occ_try1')
os.makedirs(export_root, exist_ok=True)
nusc = NuScenes(version='v1.0-mini', dataroot='misc/nuScenes/v1.0-mini/', verbose=True)
for ii in range(100):
    sample = nusc.sample[ii]

    sample_token = sample['token']
    camera_channel = 'CAM_FRONT'

    sample = nusc.get('sample', sample_token)
    lidar_token = sample['data']['LIDAR_TOP']
    camera_token = sample['data'][camera_channel]

    lidar = nusc.get('sample_data', lidar_token)
    cam = nusc.get('sample_data', camera_token)
    pcl_path = os.path.join(nusc.dataroot, lidar['filename'])
    pc = LidarPointCloud.from_file(pcl_path)
    im = Image.open(os.path.join(nusc.dataroot, cam['filename']))

    cs_record_lidar = nusc.get('calibrated_sensor', lidar['calibrated_sensor_token'])
    cs_record_cam = nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])
    poserecord_lidar = nusc.get('ego_pose', lidar['ego_pose_token'])
    poserecord_cam = nusc.get('ego_pose', cam['ego_pose_token'])
    cam_intrinsic = np.array(cs_record_cam['camera_intrinsic'])

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
    relative_pose = np.linalg.inv(cam_to_veh_mat) @ np.linalg.inv(veh_to_glb_cam_mat) @ veh_to_glb_lidar_mat @ lidar_to_veh_mat
    # Relative pose from lidar to virtual camera
    relative_pose_virtual = copy.deepcopy(relative_pose)
    relative_pose_virtual[0:3, 3] = 0
    pc_virtual = copy.deepcopy(pc)

    pc_virtual.transform(relative_pose_virtual)
    cs_record = nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])
    points = view_points(pc_virtual.points[:3, :], np.array(cs_record['camera_intrinsic']), normalize=True)
    depths = pc_virtual.points[2, :]

    min_dist = 1.0
    mask = np.ones(depths.shape[0], dtype=bool)
    mask = np.logical_and(mask, depths > min_dist)
    mask = np.logical_and(mask, points[0, :] > 0)
    mask = np.logical_and(mask, points[0, :] < im.size[0] - 1)
    mask = np.logical_and(mask, points[1, :] > 0)
    mask = np.logical_and(mask, points[1, :] < im.size[1] - 1)
    points = points[:, mask]
    depths = depths[mask]

    # plt.figure()
    # plt.imshow(im)
    # plt.scatter(points[0, :], points[1, :], c=1/depths, cmap=plt.cm.get_cmap('magma'))
    # plt.show()

    # Sparse to Dense depth map rendering
    grid_x, grid_y = np.meshgrid(range(1600),range(900))
    tri = Delaunay(points[:2,:].T)
    interp_De = LinearNDInterpolator(tri, depths, fill_value=1e5)
    intrp_d = interp_De(grid_x, grid_y)
    # plt.figure()
    # plt.imshow(1 / intrp_d, cmap=plt.cm.get_cmap('magma'))
    # plt.show()

    intrinsic = torch.eye(4, dtype=torch.float32).cuda()[None,:,:]
    intrinsic[:,:3,:3] = torch.tensor(np.array(cs_record['camera_intrinsic']), dtype=torch.float32).cuda()
    pose = relative_pose_virtual @ np.linalg.inv(relative_pose)
    pose = torch.from_numpy(pose).float().cuda().unsqueeze(0)
    depth = torch.from_numpy(intrp_d).float().unsqueeze(0).unsqueeze(0).cuda()
    occ_map = occlusion_detector.apply(intrinsic, torch.inverse(pose), depth, 1e10)

    # Sample the occlusion map
    _, _, hh, ww = occ_map.shape
    pointstorch = torch.from_numpy(points).float()
    xx, yy, _ = torch.split(pointstorch, 1, dim=0)
    xx = (xx / (ww-1) - 0.5) * 2
    yy = (yy / (hh-1) - 0.5) * 2
    occ_map_sampled = torch.nn.functional.grid_sample(occ_map.float(), torch.stack([xx, yy], dim=-1).cuda().unsqueeze(0), mode='bilinear', align_corners=False)
    occ_map_sampled = (occ_map_sampled.squeeze() > 0.8)

    pc.transform(relative_pose)
    cs_record = nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])
    points = view_points(pc.points[:3, :], np.array(cs_record['camera_intrinsic']), normalize=True)
    depths = pc.points[2, :]

    vls1path = os.path.join(export_root, 'tmp1.jpg')
    imw, imh = im.size
    plt.figure(figsize=(16, 9))
    plt.imshow(im)
    plt.scatter(points[0, mask], points[1, mask], c=1/depths[mask], cmap=plt.cm.get_cmap('magma'))
    plt.xlim([0, imw])
    plt.ylim([imh, 0])
    plt.savefig(vls1path, transparent=True, dpi=300)
    plt.close()

    vls2path = os.path.join(export_root, 'tmp2.jpg')
    mask[mask] = (occ_map_sampled.cpu().numpy() == 0)
    imw, imh = im.size
    plt.figure(figsize=(16, 9))
    plt.imshow(im)
    plt.scatter(points[0, mask], points[1, mask], c=1/depths[mask], cmap=plt.cm.get_cmap('magma'))
    plt.xlim([0, imw])
    plt.ylim([imh, 0])
    plt.savefig(vls2path, transparent=True, dpi=300)
    plt.close()

    im1 = Image.open(vls1path).resize((imw, imh))
    im2 = Image.open(vls2path).resize((imw, imh))
    im3 = tensor2disp(occ_map, vmax=1.0, viewind=0)
    im4 = tensor2disp(1 / depth, vmax=0.5, viewind=0)
    imvls1 = np.concatenate([
        np.array(im1), np.array(im2)
    ], axis=1)
    imvls2 = np.concatenate([
        np.array(im3), np.array(im4)
    ], axis=1)
    imvls = np.concatenate([imvls1, imvls2], axis=0)
    Image.fromarray(imvls).save(os.path.join(export_root, '{}.jpg'.format(str(ii))))
    # nusc.render_pointcloud_in_image(sample['token'], pointsensor_channel='LIDAR_TOP')
    a = 1