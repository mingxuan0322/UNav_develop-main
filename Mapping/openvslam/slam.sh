#!/bin/bash

work_path=/home/endeleze/Desktop/UNav_develop/Mapping
data_path=/media/endeleze/Endeleze_5T1/UNav/Mapping
script_path=$work_path/openvslam/openvslam/build
config=$work_path/openvslam/equirectangle.yaml
outf=$data_path/data/maps/$Place/$Building

[ ! -d "$outf" ] && mkdir -p "$outf"

#$script_path/run_image_slam -v $script_path/orb_vocab/orb_vocab.dbow2 -i $src_dir -c $config --no-sleep --map-db $outf/$Floor.msg > $outf/MapA.txt

$script_path/run_video_slam -v $script_path/orb_vocab/orb_vocab.dbow2 -m /home/endeleze/Desktop/UNav_develop/Mapping/data/video/Metrotech_6_Floor_4_With_Stairs.mp4 -c $config --no-sleep --map-db /home/endeleze/Desktop/UNav_develop/Mapping/data/video/$Floor.msg > $outf/MapA.txt
