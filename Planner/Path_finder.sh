#!/bin/sh

map_name=NYU_15thfloor
work_path=/home/endeleze/Desktop/WeNav
radius=1000
min_distance=5
topomap_path=$work_path/Mapping/Topomap/Output/$map_name
script=$work_path/Planner/Path_finder.py

conda activate pycolmap
python  $script --topomap_path $topomap_path --radius $radius --min_distance $min_distance
