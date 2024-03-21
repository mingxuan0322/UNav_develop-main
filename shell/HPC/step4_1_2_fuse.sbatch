#!/bin/bash

#SBATCH --job-name=4_1_2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64GB
#SBATCH --time=150:00:00
#SBATCH --gres=gpu:0
#SBATCH --output=/scratch/ay1620/WeNav/log/fs.out
#SBATCH --mail-type=END
#SBATCH --mail-user=ay1620@nyu.edu
module purge

#=========================================================
base_root=/scratch/ay1620/WeNav
mapping_root=$base_root/Mapping
work_path=$mapping_root/data/maps
#=========================================================
data_path=$mapping_root/data/src_images/$video
perspect_image_path=$data_path/perspective_images
sfm_map=$work_path/$video
output_model=$sfm_map/models
#=========================================================

dense_model=$output_model/dense_model
colmap_dir=/scratch/work/public/singularity/run-colmap-3.7.bash

$colmap_dir colmap stereo_fusion \
    --workspace_path $dense_model \
    --output_path $dense_model/$video.ply
