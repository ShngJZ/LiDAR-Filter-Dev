import os
os.environ["DGP_PATH"] = "/user/ganesang/cvl/LidarFilter/DDAD"
import os.path as osp
import sys
sys.path.append("/user/ganesang/cvl/LidarFilter/DDAD/dgp/")
sys.path.append("/user/ganesang/cvl/LidarFilter/LiDAR-Filter-Dev/")
from core.lidar_cleaner import LiDARCleaner
from dgp.datasets import SynchronizedSceneDataset
from dgp import DGP_CACHE_DIR
from tqdm import tqdm
import numpy as np
from p_tqdm import p_map
import h5py


DATA_DIR = "/scratch0/ganesang/ddad/ddad_train_val"


def geometric_transformation(rotation, translation):
    mat = np.eye(4)
    mat[:3, :3] = rotation
    mat[:3, 3] = translation
    return mat

def func(data, i=0):
    # sample, gpu_id = args
    # sample = dataset[i]
    gpu_id = i
    sample, scene_idx, sample_idx_in_scene = data
    camera_01, lidar = sample[0][0:2]
    cam_time = camera_01["timestamp"]
    lidar_time = lidar["timestamp"]

    # out_file = osp.join(OUT_DIR, f"{cam_time}@{lidar_time}.npy")

    out_file = osp.join(OUT_DIR, f"{scene_idx}@{sample_idx_in_scene}.npy")
    if osp.exists(out_file):
        return
    image_01 = camera_01['rgb']
    lidar_pts = lidar["point_cloud"]
    lidar_pts = np.concatenate([lidar_pts, np.ones_like(lidar_pts[:,[0]])], axis=1)
    w, h = image_01.size

    lidar_pose = lidar["pose"].matrix
    camera_pose = camera_01["pose"].matrix
    extrinsic = np.linalg.inv(camera_pose) @ lidar_pose
    intrinsic = camera_01["intrinsics"]

    # Translation adjustment
    x,y,z = -1.39, -0.01, -1.56
    T = geometric_transformation(np.eye(3), np.array([x, y, z]))
    inv_T = np.linalg.inv(T)


    cleaner = LiDARCleaner(
            intrinsic_cam=intrinsic[:3,:3],
            extrinsic_LiDAR2Cam=(extrinsic@inv_T)[0:3, :],
            LiDARPoints3D=(T@lidar_pts.T)[0:3, :],
            height=h, width=w,
            rszh=1, rszw=1, gpu_id=gpu_id
        )
    visible_points_filtered, lidar_to_be_occluded = cleaner(debug=False)

    
    np.save(out_file, lidar_to_be_occluded.cpu().numpy())
    data = np.load(out_file)

    # return [str(scene_idx), str(sample_idx_in_scene), str(cam_time), str(lidar_time)]
    return


# dataset = SynchronizedSceneDataset(osp.join(DATA_DIR,'ddad.json'),
#     datum_names=('lidar', 'CAMERA_01'),
#     generate_depth_from_datum='lidar',
#     split='train'
#     )

OUT_DIR = osp.join(DATA_DIR, "occ_pts")
os.makedirs(OUT_DIR, exist_ok=True)


# gpu_id_list = []
# i = 0
# while len(gpu_id_list) < len(dataset):
#     # if i == 9:
#     #     i = (i+1)%8
#     #     continue
#     gpu_id_list.append(i)
#     i = (i+1)%8


# r = p_map(func, dataset, num_cpus=os.cpu_count())
# with open("/user/ganesang/cvl/LidarFilter/DDAD/data.txt", "w") as f:
#     f.writelines([", ".join(l)+"\n" for l in r])

occ_file = h5py.File(os.path.join(DATA_DIR, 'occluded_pts.h5'), 'w')
file_list = [f.split(".")[0] for f in os.listdir(OUT_DIR)]
for file in tqdm(file_list):
    data = np.load(osp.join(OUT_DIR, f"{file}.npy"))
    occ_file[file] = data

occ_file.close()