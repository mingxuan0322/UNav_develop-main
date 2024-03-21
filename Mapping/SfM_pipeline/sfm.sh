#!/bin/bash

name=GA_01
root=/data_2/AnbangYang/WeNev
mapping_root=$root/Mapping
export localization_root=$root/Localization
bash_path=$mapping_root/SfM_pipeline/tools/one_step_sfm.sh
export netvlad_dir=$localization_root/pytorch_NetVlad
database=$mapping_root/data/src_images/$name
export netvlad_model_path=$database/netvlad_parameters/log/Aug27_00-19-26_vgg16_netvlad
export hloc_dir=$mapping_root/SfM_pipeline/Hierarchical-Localization
export image_path=$database/database
export sfm_dir=$mapping_root/data/maps/$name
export num_pairs=30
export use_netvlad="yes"
export superpoint_local="yes"
export single_camera="yes"
export use_pba="yes"
export best_match="yes"
export min_valid_ratio="0.1"
export min_match_score="0.8"
export min_num_pairs="1"
export max_try="80"
export num_seq="1"
export equirectangular="yes"
export yaw_seq=2
export GT_path=$database/Colmap_GT.mat
export GT_thre=150
export colmap_env=open3d

source $bash_path
