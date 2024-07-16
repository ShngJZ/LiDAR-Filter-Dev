# RePLAy: Removing Projective LiDAR Depthmap Artifacts via Exploiting Stereo Geometry

[Shengjie Zhu](https://shngjz.github.io)\*, 
[Girish Chandar G](girish1511.github.io)\*, 
[Abhinav Kumar](https://sites.google.com/view/abhinavkumar), 
[Xiaoming Liu](http://www.cse.msu.edu/~liuxm/index2.html)


## Setup

- Install PyTorch version compatible with your device from [here](https://pytorch.org/get-started/previous-versions/). 
- Run the following commands to install the rest of the requirements

```bash
git clone https://github.com/ShngJZ/LiDAR-Filter-Dev.git
git checkout release

pip install -r requirements.txt
```

Download the raw KITTI dataset from [here](https://www.cvlibs.net/datasets/kitti/raw_data.php) and format the directory structure as follows:

```bash
LiDAR-Cleaner
├── data
│      ├── KITTI
│      │      ├── raw
|      |      |      ├── 2011_09_26 
|      |      |      ├── 2011_09_28
|      |      |      ├── 2011_09_29
|      |      |      ├── 2011_09_30
|      |      |      └── 2011_10_03
```

## Visualization

Run the following to produce raw and clean projected depth maps overlaid on the RGB images.

```bash
python vis/vis_kitti_gen.py --data_dir "data/KITTI/raw" \
                            --seq <seq_name> \
                            --output_dir "outputs/kitti"
```

Replace `<seq_name>` with the sequence whose frames you would like the visualizations (e.g. `2011_09_26_drive_0002_sync`).

## Processed Datasets
We provide binary masks over the LiDAR scans of five datasets, namely, KITTI, KITTI360, NuScenes, Waymo and DDAD.
The masks are provided in a hdf5 format and be accessed [here](https://huggingface.co/datasets/girish1511/RePLAY). The masks can be retrieved using the following script:

```python
import h5py
import numpy as np

#Read LiDAR point cloud from respective datasets
lidar_pts = readLiDAR() 

with h5py.File("<Dataset>.h5", "r") as mask_file:
    mask = mask_file[<key>][()]

#Before projecting the LiDAR onto the camera plane, apply the mask over the LiDAR point cloud
lidar_pts = lidar_pts[(mask==0),:]
```
The `<key>`,`<Dataset>` combinations are as follows:
- KITTI and KITTI360: `<key> = "<seq_name>@<frame_number>"`. E.g., to access the mask of `2011_09_26_drive_0002_sync/0000000191.png` use `<key> = "2011_09_26_drive_0002_sync@0000000191"`
- Waymo: `<key> = "<frame_number>"`. You would need to convert Waymo to KITTI format and then use the frame numbers to access the binary masks.
- NuScenes: `<key> = `
- DDAD: `<key> = "<camera_timestamp>@<lidar_timestamp>"`. Use the [official DDAD repository](https://github.com/TRI-ML/DDAD) to read DDAD data, then the timestamps of camera and LiDAR can be retrieved as follows:

```python
from dgp.datasets import SynchronizedSceneDataset

dataset =
SynchronizedSceneDataset('<path_to_dataset>/ddad.json',
    datum_names=('lidar', 'CAMERA_01'),
    generate_depth_from_datum='lidar',
    split='train'
    )

for sample in dataset:
    lidar, camera_01 = sample[0:2]
    lidar_timestamp, camera_timestamp = lidar["timestamp"], camera_01["timestamp"] 

``` 

## Citation
If our work aided in your research, please consider starring this repo and citing:

```Bibtex
```
