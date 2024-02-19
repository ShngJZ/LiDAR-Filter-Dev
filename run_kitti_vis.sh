export DATA_DIR="/scratch1/ganesang/kitti/raw"
export SEQ="2011_09_26_drive_0002_sync"
export OUTPUT_DIR="outputs"

python vis/vis_kitti_gen.py --data_dir ${DATA_DIR} \
                            --seq ${SEQ} \
                            --output_dir ${OUTPUT_DIR}