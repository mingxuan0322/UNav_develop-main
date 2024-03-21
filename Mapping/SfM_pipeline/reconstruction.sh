#!/bin/sh
name=test1

base_root=/media/endeleze/Endeleze_5T/WeNev
mapping_root=$base_root/Mapping
localization_root=$base_root/Localization
data_path=$localization_root/pytorch_NetVlad/data/$name
image_path=$data_path/database
hloc_dir=$mapping_root/SfM_pipeline/Hierarchical-Localization
net_vlad_model=$localization_root/pytorch_NetVlad/netvlad_parameters/log/paper
work_path=$mapping_root/data/maps
slam_map=$work_path/$name.msg
sfm_map=$work_path/$name
output_model=$sfm_map/models
reconstructed_model=$output_model/reconstructed_model
refined_model=$output_model/refined_model

num_pairs=30
min_valid_ratio="0.1"
min_match_score="0.8"
min_num_pairs="1"
max_try="80"
num_seq=5
yaw_seq=3
GT_path=$data_path/Colmap_GT.mat
GT_thre=50
colmap_env=pycolmap

conda activate $colmap_env

cd $localization_root
python feature_extract.py --ckpt_path $net_vlad_model --image $image_path --output $sfm_map --superpoint_local

cd $hloc_dir
python -m hloc.match_features --best_match --global_feature_path $sfm_map/global_features.h5 --feature_path $sfm_map/feats-superpoint.h5 \
--match_output_path $sfm_map/matched.h5 --max_try $max_try --pair_file_path $sfm_map/pairs.txt --num_match_required $num_pairs \
--min_matched $min_num_pairs --min_valid_ratio $min_valid_ratio --min_match_score $min_match_score --num_seq $num_seq --equirectangular \
--yaw_seq $yaw_seq --GT_path $GT_path --GT_thre $GT_thre


python -m hloc.create_database --sfm_dir $sfm_map --image_dir $image_path --pairs $sfm_map/pairs.txt --matches $sfm_map/matched.h5 \
--features $sfm_map/feats-superpoint.h5 --slam_map $slam_map

[ ! -d "$reconstructed_model" ] && mkdir -p "$reconstructed_model"

colmap point_triangulator \
    --database_path $sfm_map/database.db \
    --image_path $image_path \
    --input_path $output_model/original_model \
    --output_path $reconstructed_model

[ ! -d "$refined_model" ] && mkdir -p "$refined_model"

colmap bundle_adjuster \
    --input_path $reconstructed_model \
    --output_path $refined_model