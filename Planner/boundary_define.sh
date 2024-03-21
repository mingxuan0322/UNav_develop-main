#!/bin/sh

work_path=/home/endeleze/Desktop/UNav_develop/Mapping/Topomap/Output
script=/home/endeleze/Desktop/UNav_develop/Planner/boundary_define.py

source ~/anaconda3/etc/profile.d/conda.sh

conda activate pycolmap
python $script --work_path $work_path --Place $Place --Building $Building --Floor $Floor --plan $mapping_root/data/floor_plan/$Place/$Building/$Floor
