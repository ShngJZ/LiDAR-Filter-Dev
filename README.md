# RePLAy: Removing Projective LiDAR Depthmap Artifacts via Exploiting Stereo Geometry

This repository contains the official implementation for our ECCV 2024 paper on LiDAR depthmap artifact removal using stereo geometry.

## 📝 Paper Information
**Title:** RePLAy: Remove Projective LiDAR Depthmap Artifacts via Exploiting Epipolar Geometry  
**Authors:**  
[Shengjie Zhu](https://shngjz.github.io)*,  
[Girish Chandar G](https://girish1511.github.io)*,  
[Abhinav Kumar](https://sites.google.com/view/abhinavkumar),  
[Xiaoming Liu](http://www.cse.msu.edu/~liuxm/index2.html)  

🔗 [arXiv preprint](https://arxiv.org/abs/2407.19154) | 🌐 [Project Page](https://shngjz.github.io/RePLAy/)

## 🛠️ Setup

### Installation
1. Install PyTorch from the [official website](https://pytorch.org/get-started/locally/)
2. Install remaining dependencies:
```bash
pip install -r requirements.txt
```

## 🚀 Usage

### Precomputed Masks
We provide binary masks for LiDAR scans from five popular datasets:
- KITTI
- KITTI360
- NuScenes
- Waymo
- DDAD

The masks are available in HDF5 format on [Hugging Face](https://huggingface.co/datasets/girish1511/RePLAY).

### Quick Start
1. Download the precomputed masks:
```bash
git clone https://huggingface.co/datasets/girish1511/RePLAy
```

2. Visualize Pre-computed Masks Data Storage Structure
```bash
python demo/display_h5_structure.py \
    --data-root [your-downloaded-precomputed-lidar-masks] \
    --dataset [kitti or kitti360 or nuscenes or waymo or ddad]
```

3. Visualize KITTI LiDAR pointcloud cleaning:
```bash
python demo/kitt/visualization.py \
    --kitti_root [your-kitti-data-root] \
    --occlusion_root [your-downloaded-precomputed-lidar-masks] \
    --demo_dir [where-to-save-demo-visualization]
```

4. Generate cleaned KITTI LiDAR pointclouds:
```bash
# First download KITTI data
./misc/raw_kitti_data_downloader.sh

# Then run filtering
python demo/kitti/generation.py \
    --data_dir [your-kitti-data-root] \
    --seq [your-interested-kitti-sequence] \
    --output_dir [where-to-save-output]
```

## 📜 Citation
If you find our work useful for your research, please consider citing:
```bibtex
@inproceedings{zhu2024replay,
  title={RePLAy: Remove Projective LiDAR Depthmap Artifacts via Exploiting Epipolar Geometry},
  author={Zhu, Shengjie and Ganesan, Girish Chandar and Kumar, Abhinav and Liu, Xiaoming},
  booktitle={ECCV},
  year={2024},
}
```

## 🤝 Contributing
We welcome contributions! Please open an issue or submit a pull request for any improvements.