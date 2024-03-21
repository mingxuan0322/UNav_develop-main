#!/bin/sh

video=GA_01

base_root=/home/endeleze/Desktop/WeNav
mapping_root=$base_root/Mapping
topo_root=$mapping_root/Topomap
topo_out=$topo_root/Output/$video
work_path=$mapping_root/data/maps
sfm_map=$work_path/$video
output_model=$sfm_map/models
reconstructed_model=$output_model/reconstructed_model
colmap2topomap_script=$topo_root/colmap2topometric.py

source ~/anaconda3/etc/profile.d/conda.sh
conda activate pycolmap
python $colmap2topomap_script --outf $topo_out --maps $reconstructed_model