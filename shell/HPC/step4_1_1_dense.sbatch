#!/bin/bash

#SBATCH --job-name=4_1_1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=64GB
#SBATCH --time=150:00:00
#SBATCH --gres=gpu:3
#SBATCH --output=/scratch/ay1620/WeNav/log/ds.out
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
reconstructed_model=$output_model/reconstructed_model
refined_model=$output_model/refined_model
#=========================================================

dense_model=$output_model/dense_model
colmap_dir=/scratch/work/public/singularity/run-colmap-3.7.bash

$colmap_dir colmap image_undistorter \
    --image_path $perspect_image_path \
    --input_path $refined_model \
    --output_path $dense_model

$colmap_dir colmap patch_match_stereo \
    --workspace_path $dense_model
