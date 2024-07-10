# Removing Projective LiDAR Depthmap Artifacts via Exploiting Stereo Geometry

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

## Citation
If our work aided in your research, please consider starring this repo and citing:

```Bibtex
```
