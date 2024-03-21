#!/bin/sh

Place=New_York_City
Building=LightHouse
Floor=3_

#=========================================================
work_root=/home/endeleze/Desktop/UNav_develop
base_root=/media/endeleze/Endeleze_5T1/UNav
mapping_root=$base_root/Mapping
topo_root=$mapping_root/Topomap
topo_out=$topo_root/Output/$Place/$Building/$Floor
localization_root=$base_root/Localization
planner_root=$base_root/Planner
work_path=$mapping_root/data/maps
#=========================================================
data_path=$mapping_root/data/src_images/$Place/$Building/$Floor
equirect_image_path=$data_path/equirectangular_images
perspect_image_path=$data_path/perspective_images
hloc_dir=$mapping_root/SfM_pipeline/Hierarchical-Localization
net_vlad_model=$localization_root/pytorch_NetVlad/netvlad_parameters/log/paper
slam_map=$work_path/$Place/$Building/$Floor.msg
sfm_map=$work_path/$Place/$Building/$Floor
output_model=$sfm_map/models
reconstructed_model=$output_model/reconstructed_model
dense_model=$output_model/dense_model
#=========================================================
equi2pers_script=$work_root/Localization/Equirec2Perspec.py
db_Topo_path=$topo_out/slam_data.json
db_pitch_num=1
db_pitch_range=20
db_yaw_num=18
db_FOV=75
frame_width=640
frame_height=360
frame_skip=1

num_pairs=50
min_valid_ratio="0.1"
min_match_score="0.8"
min_num_pairs="1"
max_try="80"
num_seq=5
yaw_seq=3
GT_path=$data_path/Colmap_GT.mat
GT_thre=50
batch_size=10
#=========================================================
colmap2topomap_script=$work_root/Topomap/colmap2topometric.py
path_script=$work_root/Planner/Path_finder.py
radius=1000
min_distance=5
#=========================================================
colmap_env=pycolmap

conda activate $colmap_env

python  $equi2pers_script --root $data_path --db_src_root $equirect_image_path --db_Topo_path $db_Topo_path \
--db_pitch_num $db_pitch_num --db_yaw_num $db_yaw_num --db_pitch_range $db_pitch_range --db_FOV $db_FOV \
--frame_width $frame_width --frame_height $frame_height --frame_skip $frame_skip \
--dataset $Building/$Floor

#cd $localization_root
#python feature_extract.py --ckpt_path $net_vlad_model --image $perspect_image_path --output $sfm_map --superpoint_local
#
#cd $hloc_dir
#python -m hloc.best_matches --global_feature_path $sfm_map/global_features.h5 --feature_path $sfm_map/feats-superpoint.h5 \
#--match_output_path $sfm_map/matched.h5 --max_try $max_try --pair_file_path $sfm_map/pairs.txt --num_match_required $num_pairs \
#--min_matched $min_num_pairs --min_valid_ratio $min_valid_ratio --min_match_score $min_match_score --num_seq $num_seq \
#--yaw_seq $yaw_seq --GT_path $GT_path --GT_thre $GT_thre --batch_size $batch_size
#
#python -m hloc.reconstruct --sfm_dir $sfm_map --image_dir $perspect_image_path --pairs $sfm_map/pairs.txt --matches $sfm_map/matched.h5 \
#--features $sfm_map/feats-superpoint.h5 --slam_map $slam_map
#
#python $colmap2topomap_script --outf $topo_out --maps $reconstructed_model
#
#python  $path_script --topomap_path $topo_out --radius $radius --min_distance $min_distance
#cd $base_root/shell
#
#colmap image_undistorter \
#    --image_path $perspect_image_path \
#    --input_path $reconstructed_model \
#    --output_path $dense_model
#
#colmap patch_match_stereo \
#    --workspace_path $dense_model
#
#colmap stereo_fusion \
#    --workspace_path $dense_model \
#    --output_path $dense_model/fused.ply
