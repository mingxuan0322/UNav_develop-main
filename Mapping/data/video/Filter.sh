#!/bin/sh

video=NYU_15thfloor

work_path=/home/endeleze/Desktop/WeNev/Mapping/data
src=$work_path/video/$video.mp4
export src_dir=$work_path/src_images/$video
rate=10
scale=-1:1940

[ ! -d "$src_dir" ] && mkdir -p "$src_dir"

ffmpeg -i $src -filter_complex \
"[0:v]trim=start=80:100,setpts=PTS-STARTPTS[a]; \
[0:v]trim=start=100:110,setpts=PTS-STARTPTS[b]; \
[0:v]trim=start=110:120,setpts=PTS-STARTPTS[c]; \
[0:v]trim=start=120:200,setpts=PTS-STARTPTS[d]; \
[a][b]concat[e1];\
[e1][c]concat[e2];\
[e2][d]concat[e3]" -map [e3] test.mp4


