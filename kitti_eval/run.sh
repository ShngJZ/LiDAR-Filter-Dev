export DATA_PATH="/scratch1/ganesang/kitti/datasets"

#'semidense_file' is the split file to use
#'test-depth' denotes raw/clean depth points to verify
python main.py --data-path $DATA_PATH --semidense-file $2 --test-depth $1