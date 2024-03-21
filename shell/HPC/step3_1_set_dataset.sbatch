#!/bin/bash

#SBATCH --job-name=3_1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=200GB
#SBATCH --time=150:00:00
#SBATCH --gres=gpu:0
#SBATCH --output=/scratch/ay1620/WeNav/log/sd.out
#SBATCH --mail-type=END
#SBATCH --mail-user=ay1620@nyu.edu
module purge

#=========================================================
base_root=/scratch/ay1620/WeNav
mapping_root=$base_root/Mapping
localization_root=$base_root/Localization
work_path=$mapping_root/data/maps
#=========================================================
data_path=$mapping_root/data/src_images/$video
perspect_image_path=$data_path/perspective_images
net_vlad_model=$localization_root/pytorch_NetVlad/netvlad_parameters/log/paper
slam_map=$work_path/$video.msg
sfm_map=$work_path/$video
#=========================================================

num_pairs=50
min_valid_ratio="0.1"
min_match_score="0.8"
min_num_pairs="1"
max_try="80"
num_seq=5
yaw_seq=3
GT_path=$data_path/Colmap_GT.mat
GT_thre=1000
batch_size=5
#=========================================================

overlay=/scratch/work/public/singularity/colmap-3.7-cuda10.1-cudnn7-devel-ubuntu18.04-20201207.sqf
singularity exec --nv \
            --overlay /scratch/ay1620/WeNav/50G.ext3:ro \
	    --overlay $overlay:ro \
	    /scratch/ay1620/WeNav/cuda10.sif \
            /bin/bash -c "
source /ext3/env.sh;
cd /scratch/ay1620/WeNav/Mapping/SfM_pipeline/Hierarchical-Localization;
python -m hloc.setup_dataset --sfm_dir $sfm_map --image_dir $perspect_image_path --pairs $sfm_map/pairs.txt --matches $sfm_map/matched.h5 \
--features $sfm_map/feats-superpoint.h5 --slam_map $slam_map"
