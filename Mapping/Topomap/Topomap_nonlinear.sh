#!/bin/sh

data_root=/media/endeleze/Endeleze_5T1/UNav/Mapping
work_path=/home/endeleze/Desktop/UNav_develop/Mapping

script=$work_path/Topomap/Topomap_nonlinear.py
maps=$data_root/data/maps/$Place/$Building/$Floor.msg
outf=$data_root/Topomap/Output/$Place/$Building/$Floor
plan=$data_root/data/floor_plan

conda activate pycolmap
python  $script --maps $maps --src_dir $src_dir --outf $outf --plan $plan
