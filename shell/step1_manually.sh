#!/bin/bash

export Place=New_York_City
export Building=LightHouse
export Floor=6
export rate=30

data_root=/media/endeleze/Endeleze_5T1/UNav
work_path=/home/endeleze/Desktop/UNav_develop
mapping_root=$work_path/Mapping
planner_root=$work_path/Planner
boundary_script=$planner_root/boundary_define.py
#export src_dir=$data_root/Mapping/data/src_images/$Place/$Building/$Floor/equirectangular_images
export src_dir=$work_path/Mapping/data/src_images/$Place/$Building/$Floor/equirectangular_images

#source $mapping_root/data/extract.sh
#source $mapping_root/openvslam/slam.sh
#source $mapping_root/Topomap/Topomap_nonlinear.sh

conda activate pycolmap
source $planner_root/boundary_define.sh

