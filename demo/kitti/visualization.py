import argparse
import glob
import os
import shutil

import h5py
import matplotlib.pyplot as plt
import natsort
import numpy as np
import tqdm
from PIL import Image
def read_calib_file(path):
    """Read KITTI calibration file (from https://github.com/hunse/kitti)
    Args:
        path: Path to calibration file
    Returns:
        dict: Parsed calibration data with float arrays where possible
    """
    float_chars = set("0123456789.e+- ")
    data = {}
    with open(path, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            value = value.strip()
            data[key] = value
            if float_chars.issuperset(value):
                try:
                    data[key] = np.array(list(map(float, value.split(' '))))
                except ValueError:
                    pass  # Keep original string value if conversion fails
    return data

def acquire_kitti_seqs(kitti_root):
    """Scan KITTI dataset directory structure to find all sequences
    Args:
        kitti_root: Path to KITTI dataset root directory
    Returns:
        list: Sorted list of [date, sequence] pairs
    """
    dates = [os.path.basename(x) for x in glob.glob(os.path.join(kitti_root, "*"))]
    seqs = []
    for date in dates:
        date_folder = os.path.join(kitti_root, date)
        if os.path.isdir(date_folder):
            seqs.extend([
                [date, os.path.basename(x)]
                for x in glob.glob(os.path.join(date_folder, "*"))
                if os.path.isdir(x)
            ])
    return seqs

def plot_kitti(seq_time, rgb, x, y, z, valid, legend):
    """Plot LiDAR points on RGB image with depth coloring
    Args:
        seq_time: Sequence timestamp for filename
        rgb: PIL Image object
        x,y,z: LiDAR point coordinates in image plane
        valid: Boolean mask for valid points
        legend: Text legend to display on image
    Returns:
        PIL.Image: Rendered visualization image
    """
    cm = plt.get_cmap('magma')
    color = cm(6 / z[valid])
    tmp_path = f'{seq_time}.jpg'

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
    """Process KITTI sequence to generate visualization videos
    Args:
        kitti_root: Path to KITTI dataset
        occlusion_root: Path to occlusion data
        demo_dir: Output directory
        seq_date: Date folder name
        seq_time: Sequence folder name
        cam: Camera ID to use (default: 2)
    """
    demo_dir_seq = os.path.join(demo_dir, seq_time)
    if os.path.exists(demo_dir_seq):
        return

    os.makedirs(demo_dir_seq, exist_ok=True)
    video_path = os.path.join(demo_dir, f"{seq_time}.mp4")
    if os.path.exists(video_path):
        return

    # Path setup
    LIDAR_DIR = os.path.join(kitti_root, seq_date, seq_time, 'velodyne_points/data')
    CAM_CALIB_PATH = os.path.join(kitti_root, seq_date, 'calib_cam_to_cam.txt')
    LIDAR_CALIB_PATH = os.path.join(kitti_root, seq_date, 'calib_velo_to_cam.txt')

    # Load calibration data
    cam2cam = read_calib_file(CAM_CALIB_PATH)
    velo2cam = read_calib_file(LIDAR_CALIB_PATH)

    # Construct transformation matrices
    velo2cam = np.hstack((velo2cam['R'].reshape(3, 3), velo2cam['T'][..., np.newaxis]))
    velo2cam = np.vstack((velo2cam, np.array([0, 0, 0, 1.0])))

    R_cam2rect = np.eye(4)
    R_cam2rect[:3, :3] = cam2cam['R_rect_00'].reshape(3, 3)
    P_rect = cam2cam[f'P_rect_0{cam}'].reshape(3, 4)
    P_velo2im = np.dot(np.dot(P_rect, R_cam2rect), velo2cam)

    # Process each frame
    h5file = h5py.File(os.path.join(occlusion_root, "kitti.h5"), 'r')
    imgs = natsort.natsorted(glob.glob(
        os.path.join(kitti_root, seq_date, seq_time, f"image_0{cam}/data", "*.png")
    ))

    for imgname in tqdm.tqdm(imgs):
        frm_num = os.path.basename(imgname).split('.')[0]

        # Load LiDAR data
        lidar_pts = np.fromfile(
            os.path.join(LIDAR_DIR, f'{frm_num}.bin'),
            dtype=np.float32
        ).reshape((-1, 4))
        lidar_pts[:, 3] = 1.0  # Homogeneous coordinates

        # Load occlusion data
        occlusion = np.array(h5file[f"{seq_time}@{frm_num}"])
        rgb = Image.open(imgname)
        w, h = rgb.size

        # Project LiDAR to image
        lidar_pts_im = np.dot(P_velo2im, lidar_pts.T).T
        x = lidar_pts_im[:, 0] / lidar_pts_im[:, 2]
        y = lidar_pts_im[:, 1] / lidar_pts_im[:, 2]
        z = lidar_pts_im[:, 2]

        # Validity masks
        valid_wo_replay = (x > 0) * (x < w-1) * (y > 0) * (y < h-1) * (z > 0)
        valid_wt_replay = valid_wo_replay * (occlusion == 0)

        # Generate visualizations
        rgb_wo = plot_kitti(seq_time, rgb, x, y, z, valid_wo_replay, "Raw")
        rgb_wt = plot_kitti(seq_time, rgb, x, y, z, valid_wt_replay, "Ours")

        # Save combined result
        combined = np.concatenate([np.array(rgb_wo), np.array(rgb_wt)], axis=0)
        Image.fromarray(combined).save(os.path.join(demo_dir_seq, f"{frm_num}.jpg"))

    # Generate video and clean up
    os.system(
        f"ffmpeg -framerate 5 -pattern_type glob -i '{demo_dir_seq}/*.jpg' "
        f"-c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p {video_path}"
    )
    shutil.rmtree(demo_dir_seq, ignore_errors=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KITTI Visualization Tool")
    parser.add_argument("--kitti_root", default="/home/ubuntu/disk5/KITTI",
                       help="Path to KITTI dataset root")
    parser.add_argument("--occlusion_root", default="/home/ubuntu/disk5/RePLAy",
                       help="Path to occlusion data")
    parser.add_argument("--demo_dir", default="/home/ubuntu/disk5/LiDAR-Filter-Dev/demooutput",
                       help="Output directory for visualizations")

    args = parser.parse_args()
    demo_kitti_root = os.path.join(args.demo_dir, "KITTI")
    os.makedirs(demo_kitti_root, exist_ok=True)

    # Process first sequence only (for demo)
    seq_date, seq_time = natsort.natsorted(acquire_kitti_seqs(args.kitti_root))[0]
    clean_kitti(args.kitti_root, args.occlusion_root, demo_kitti_root, seq_date, seq_time)