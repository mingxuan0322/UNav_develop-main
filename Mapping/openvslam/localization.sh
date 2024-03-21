#!/bin/bash

export Place=New_York_City
export Building=LightHouse
export Floor_B=3_
export Floor_A=3
export rate=20

data_root=/media/endeleze/Endeleze_5T1/UNav
work_path=/home/endeleze/Desktop/UNav_develop
mapping_root=$work_path/Mapping
planner_root=$work_path/Planner
boundary_script=$planner_root/boundary_define.py
config=$mapping_root/openvslam/equirectangle.yaml
script_path=$mapping_root/openvslam/openvslam/build
outf=$data_root/Mapping/data/maps/$Place/$Building
export src_dir=$data_root/Mapping/data/src_images/$Place/$Building/$Floor_B/equirectangular_images

#source $mapping_root/data/extract_B.sh

$script_path/run_image_localization -v $script_path/orb_vocab/orb_vocab.dbow2 -i $src_dir -c $config --frame-skip 1 --no-sleep -p $outf/$Floor_A.msg > $outf/mapAB.txt
