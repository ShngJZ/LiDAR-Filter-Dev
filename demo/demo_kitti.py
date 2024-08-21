import random

import natsort, os, glob, h5py, shutil, tqdm
import numpy as np
import matplotlib.pyplot as plt
import argparse

from PIL import Image

def read_calib_file(path):
    """Read KITTI calibration file
    (from https://github.com/hunse/kitti)
    """
    float_chars = set("0123456789.e+- ")
    data = {}
    with open(path, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            value = value.strip()
            data[key] = value
            if float_chars.issuperset(value):
                # try to cast to float array
                try:
                    data[key] = np.array(list(map(float, value.split(' '))))
                except ValueError:
                    # casting error: data[key] already eq. value, so pass
                    pass

    return data


def acquire_kitti_seqs(kitti_root):
    dates = [os.path.basename(x) for x in glob.glob(os.path.join(args.kitti_root, "*"))]
    seqs = list()
    for date in dates:
        date_folder = os.path.join(kitti_root, date)
        if os.path.isdir(date_folder):
            for x in glob.glob(os.path.join(date_folder, "*")):
                if os.path.isdir(x):
                    seqs.append([date, os.path.basename(x)])
    return seqs

def plot_kitti(seq_time, rgb, x, y, z, valid, legend):
    cm = plt.get_cmap('magma')
    color = cm(6 / z[valid])
    tmp_path = '{}.jpg'.format(seq_time)
    w, h = rgb.size
    plt.figure(figsize=(16, 9))
    plt.imshow(rgb, alpha=0.4)
    plt.scatter(x[valid], y[valid], s=2.0, c=color, marker='.')
    plt.axis('off')
    plt.xlim([0, w])
    plt.ylim([h, 0])
    t = plt.text(10, 30, legend, fontsize=20)
    t.set_bbox(dict(facecolor='white', alpha=0.2))
    plt.savefig(tmp_path, transparent=True, bbox_inches='tight', dpi=300, pad_inches=0)
    plt.close()
    plotted = Image.open(tmp_path).resize((w, h))
    os.remove(tmp_path)
    return plotted

def clean_kitti(kitti_root, occlusion_root, demo_dir, seq_date, seq_time, cam=2):
    demo_dir_seq = os.path.join(demo_dir, seq_time)
    if os.path.exists(demo_dir_seq):
        return
    os.makedirs(demo_dir_seq, exist_ok=True)
    video_path = os.path.join(demo_dir, "{}.mp4".format(seq_time))
    if os.path.exists(video_path):
        return

    LIDAR_DIR = os.path.join(kitti_root, seq_date, seq_time, 'velodyne_points/data')
    CAM_CALIB_PATH = os.path.join(kitti_root, seq_date, 'calib_cam_to_cam.txt')
    LIDAR_CALIB_PATH = os.path.join(kitti_root, seq_date, 'calib_velo_to_cam.txt')

    cam2cam = read_calib_file(CAM_CALIB_PATH)
    velo2cam = read_calib_file(LIDAR_CALIB_PATH)
    velo2cam = np.hstack((velo2cam['R'].reshape(3, 3), velo2cam['T'][..., np.newaxis]))
    velo2cam = np.vstack((velo2cam, np.array([0, 0, 0, 1.0])))

    R_cam2rect = np.eye(4)
    R_cam2rect[:3, :3] = cam2cam['R_rect_00'].reshape(3, 3)
    P_rect = cam2cam['P_rect_0'+str(cam)].reshape(3, 4)
    P_velo2im = np.dot(np.dot(P_rect, R_cam2rect), velo2cam)

    h5file_path = os.path.join(occlusion_root, "kitti.h5")
    h5file = h5py.File(h5file_path, 'r')

    imgs = glob.glob(os.path.join(args.kitti_root, seq_date, seq_time, "image_0{}/data".format(str(cam)), "*.png"))
    imgs = natsort.natsorted(imgs)
    for imgname in tqdm.tqdm(imgs):
        frm_num = os.path.basename(imgname).split('.')[0]
        lidar_pts = np.fromfile(os.path.join(LIDAR_DIR, '{}.bin'.format(frm_num)), dtype=np.float32).reshape((-1, 4))
        lidar_pts[:, 3] = 1.0

        occlusion = np.array(h5file["{}@{}".format(seq_time, frm_num)])

        rgb = Image.open(imgname)
        w, h = rgb.size

        lidar_pts_im = np.dot(P_velo2im, lidar_pts.T).T
        x, y, z = lidar_pts_im[:, 0] / lidar_pts_im[:, 2], lidar_pts_im[:, 1] / lidar_pts_im[:, 2], lidar_pts_im[:, 2]
        valid_wo_replay = (x > 0) * (x < w-1) * (y > 0) * (y < h-1) * (z > 0)
        valid_wt_replay = (x > 0) * (x < w-1) * (y > 0) * (y < h-1) * (z > 0) * (occlusion == 0)

        rgb_wo = plot_kitti(seq_time, rgb, x, y, z, valid_wo_replay, legend="Raw")
        rgb_wt = plot_kitti(seq_time, rgb, x, y, z, valid_wt_replay, legend="Ours")

        combined = np.concatenate([
            np.array(rgb_wo),
            np.array(rgb_wt)
        ], axis=0)

        Image.fromarray(combined).save(os.path.join(demo_dir_seq, "{}.jpg".format(frm_num)))

    generate_video = "ffmpeg -framerate 5 -pattern_type glob -i '{}/*.jpg' -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p {}".format(demo_dir_seq, video_path)
    os.system(generate_video)
    if os.path.exists(demo_dir_seq):
        shutil.rmtree(demo_dir_seq)
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Testing", conflict_handler="resolve")

    parser.add_argument("--kitti_root", default="/media/shengjie/scratch2/RePLAy/Data/KITTI")
    parser.add_argument("--occlusion_root", default="/media/shengjie/scratch2/RePLAy/Occlusion")
    parser.add_argument("--demo_dir", default="/media/shengjie/scratch2/RePLAy/Demos")

    args = parser.parse_args()

    demo_kitti_root = os.path.join(args.demo_dir, "KITTI")
    os.makedirs(demo_kitti_root, exist_ok=True)
    seqs = acquire_kitti_seqs(args.kitti_root)
    random.shuffle(seqs)

    for seq_date, seq_time in seqs:
        clean_kitti(args.kitti_root, args.occlusion_root, demo_kitti_root, seq_date, seq_time)
        