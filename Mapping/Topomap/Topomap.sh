#!/bin/sh

work_path=/media/endeleze/Endeleze_5T/UNav/Mapping
script=$work_path/Topomap/Topomap.py
maps=$work_path/data/maps/$Place/$Building/$Floor.msg
outf=$work_path/Topomap/Output/$Place/$Building/$Floor
plan=$work_path/data/floor_plan

conda activate pycolmap
python  $script --maps $maps --src_dir $src_dir --outf $outf --plan $plan
