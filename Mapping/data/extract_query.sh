#!/bin/sh
video=query_NYU17
work_path=/home/endeleze/Desktop/WeNav/Mapping/data
src=$work_path/video/$video.mp4
export src_dir=$work_path/src_images/$video
rate=4
scale=640:-1
[ ! -d "$src_dir" ] && mkdir -p "$src_dir"

#ffmpeg -i $src -r $rate $src_dir/%04d.png
ffmpeg -i $src -r $rate -vf scale=$scale $src_dir/%04d.png
