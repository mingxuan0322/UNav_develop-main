#!/bin/sh

db_video=GA_01
q_video=GA_01_sparse

base_root=/media/endeleze/Endeleze_5T/WeNev
root=$base_root/Localization/pytorch_NetVlad/data/$db_video

db_input=$base_root/Mapping/data/src_images/$db_video
db_Topo_path=$base_root/Mapping/Topomap/Output/$db_video/topo-map.json
db_pitch_num=1
db_pitch_range=20
db_yaw_num=18
db_FOV=75

q_input=$base_root/Mapping/data/src_images/$q_video
q_Topo_path=$base_root/Mapping/Topomap/Output/$q_video/topo-map.json
q_pitch_num=1
q_pitch_range=20
q_yaw_num=12
q_FOV=60

valid_ratio=0.05

frame_width=640
frame_height=360
frame_skip=1
posDistThr=25
posDistSqThr=625
nonTrivPosDistSqThr=100

script=$base_root/Localization/Equirec2Perspec.py
source ~/anaconda3/etc/profile.d/conda.sh

conda activate open3d

python  $script --root $root --db_src_root $db_input --db_Topo_path $db_Topo_path \
--db_pitch_num $db_pitch_num --db_yaw_num $db_yaw_num --db_pitch_range $db_pitch_range --db_FOV $db_FOV \
--frame_width $frame_width --frame_height $frame_height --frame_skip $frame_skip \
--posDistThr $posDistThr --posDistSqThr $posDistSqThr --nonTrivPosDistSqThr $nonTrivPosDistSqThr --valid_ratio $valid_ratio --dataset $db_video \
#--q_pitch_num $q_pitch_num --q_pitch_range $q_pitch_range --q_yaw_num $q_yaw_num --q_FOV $q_FOV --q_src_root $q_input --q_Topo_path $q_Topo_path
