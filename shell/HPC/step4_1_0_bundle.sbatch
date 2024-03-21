#!/bin/bash

#SBATCH --job-name=4_1_0
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=64GB
#SBATCH --time=150:00:00
#SBATCH --gres=gpu:0
#SBATCH --output=/scratch/ay1620/WeNav/log/rf.out
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
[ ! -d "$refined_model" ] && mkdir -p "$refined_model"

colmap_dir=/scratch/work/public/singularity/run-colmap-3.7.bash

$colmap_dir colmap bundle_adjuster --input_path $reconstructed_model --output_path $refined_model

for i in {1..49}
do
$colmap_dir colmap bundle_adjuster --input_path $refined_model --output_path $refined_model
done

