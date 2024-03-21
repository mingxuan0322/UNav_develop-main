#!/bin/sh

work_path=/media/endeleze/Endeleze_5T1/UNav/Mapping/data
#src=$work_path/video/$Place/$Building/$Floor.mp4
src=/home/endeleze/Desktop/UNav_develop/Mapping/data/video/Metrotech_6_Floor_4_With_Stairs.mp4
[ ! -d "$src_dir" ] && mkdir -p "$src_dir"

ffmpeg -i $src -r $rate $src_dir/%05d.png 
