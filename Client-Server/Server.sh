#!/bin/bash

Place=New_York_University
Building=NYU_Langone
Floor=17_DENSE_LOW
FloorPlan_scale=0.01

work_path=/media/endeleze/Endeleze_5T/UNav

topomap_path=$work_path/Mapping/Topomap/Output
database_path=$work_path/Mapping/data/maps
logs=$work_path/Localization/trials.json
ckpt_path=$work_path/Localization/pytorch_NetVlad/netvlad_parameters/log/paper
max_matches=20
max_matching_image_num=100

host_id=128.122.136.119
port_id=30002
script=$work_path/Client-Server/Server.py

source ~/anaconda3/etc/profile.d/conda.sh
conda activate pycolmap

python $script --host_id $host_id --port_id $port_id \
--ckpt_path $ckpt_path --FloorPlan_scale $FloorPlan_scale \
--Place $Place --Building $Building --Floor $Floor --topomap_path $topomap_path --database_path $database_path --logs $logs \
--cpu --max_matches $max_matches --max_matching_image_num $max_matching_image_num
