#!/bin/bash

Place=New_York_University
Building=NYU_Langone
Floor=17_DENSE_LOW
query_name=$Place/$Building/$Floor

work_path=/home/endeleze/Desktop/UNav_develop
# data_path=/media/endeleze/Endeleze_5T/UNav
data_path=/home/endeleze/Desktop/UNav_develop
topomap_path=$data_path/Mapping/Topomap/Output/$Place/$Building/$Floor
GT_path=$data_path/Mapping/Topomap/Output/$query_name
db_root=$data_path/Mapping/data/maps/$Place/$Building/$Floor
fl_path=$data_path/Mapping/data/floor_plan
global_descriptor=$db_root/global_features.h5
local_descriptor=$db_root/feats-superpoint.h5
path_path=$topomap_path/path.h5
db_dir=$data_path/Mapping/data/src_images/$Place/$Building/$Floor/perspective_images
query_dir=$data_path/Mapping/data/src_images/$query_name
ckpt_path=$data_path/Localization/pytorch_NetVlad/netvlad_parameters/log/paper

source ~/anaconda3/etc/profile.d/conda.sh

conda activate pycolmap
python $work_path/Localization/localization.py --db_dir $db_dir --fl_path $fl_path --query_dir $query_dir --ckpt_path $ckpt_path \
--global_descriptor $global_descriptor --local_descriptor $local_descriptor --topomap_path $topomap_path --GT_path $GT_path --path_path $path_path \
--cpu
