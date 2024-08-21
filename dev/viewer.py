import sys
# walk directories
import glob
# access to OS functionality
import os
# copy things
import copy
# numpy
import numpy as np

class Kitti360Viewer3DRaw(object):

    # Constructor
    def __init__(self, seq=0, mode='velodyne', BASE_DIR="/scratch1/ganesang/kitti360/"):

        kitti360Path = BASE_DIR

        if mode=='velodyne':
            self.sensor_dir='velodyne_points'
        elif mode=='sick':
            self.sensor_dir='sick_points'
        else:
            raise RuntimeError('Unknown sensor type!')

        sequence = '2013_05_28_drive_%04d_sync' % seq
        self.raw3DPcdPath  = os.path.join(kitti360Path, 'data_3d_raw', sequence, self.sensor_dir, 'data')

    def loadVelodyneData(self, frame=0):
        pcdFile = os.path.join(self.raw3DPcdPath, '%010d.bin' % frame)
        if not os.path.isfile(pcdFile):
            raise RuntimeError('%s does not exist!' % pcdFile)
        pcd = np.fromfile(pcdFile, dtype=np.float32)
        pcd = np.reshape(pcd,[-1,4])
        return pcd 

    def loadSickData(self, frame=0):
        pcdFile = os.path.join(self.raw3DPcdPath, '%010d.bin' % frame)
        if not os.path.isfile(pcdFile):
            raise RuntimeError('%s does not exist!' % pcdFile)
        pcd = np.fromfile(pcdFile, dtype=np.float32)
        pcd = np.reshape(pcd,[-1,2])
        pcd = np.concatenate([np.zeros_like(pcd[:,0:1]), -pcd[:,0:1], pcd[:,1:2]], axis=1)
        return pcd

def projectVeloToImage(cam_id=0, seq=0, frame=0, BASE_DIR="/scratch1/ganesang/kitti360/"):
    from loadCalibration import loadCalibrationCameraToPose, loadCalibrationRigid
    from camera_helper import CameraPerspective, CameraFisheye
    from PIL import Image
    import matplotlib.pyplot as plt

    kitti360Path = BASE_DIR
    
    sequence = seq#'2013_05_28_drive_%04d_sync'%seq

    # perspective camera
    if cam_id in [0,1]:
        camera = CameraPerspective(kitti360Path, sequence, cam_id)
    # fisheye camera
    elif cam_id in [2,3]:
        camera = CameraFisheye(kitti360Path, sequence, cam_id)
    else:
        raise RuntimeError('Unknown camera ID!')

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
    cm = plt.get_cmap('jet')

    # visualize a set of frame
    # for each frame, load the raw 3D scan and project to image plane
    # for frame in range(0, 1000, 50):
    
    points = velo.loadVelodyneData(frame)
    points[:,3] = 1

    # transfrom velodyne points to camera coordinate
    pure_rot = TrVeloToRect.copy()
    pure_rot[:3,-1] = 0.0
    pointsCam = np.matmul(TrVeloToRect, points.T).T
    pointsCam = pointsCam[:,:3]
    # project to image space
    u,v, depth= camera.cam2image(pointsCam.T)
    
    u = u.astype(np.int32)
    v = v.astype(np.int32)

    # prepare depth map for visualization
    depthMap = np.zeros((camera.height, camera.width))
    mask = np.logical_and(np.logical_and(np.logical_and(u>=0, u<camera.width), v>=0), v<camera.height)
    mask = np.logical_and(mask, depth>0)
    depthMap[v[mask],u[mask]] = depth[mask]
    return depthMap