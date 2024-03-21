#!/bin/sh

video=GS020012

work_path=/home/endeleze/Desktop/WeNev
image_dir=$work_path/Mapping/data/src_images/$video
hfnet_path=$work_path/Localization/hfnet
outf=$hfnet_path/Output
script_path=$hfnet_path/hfnet_features_extract.py

source ~/anaconda3/etc/profile.d/conda.sh
conda activate BELL
python $script_path --image_dir $image_dir --output $outf --model_dir $hfnet_path
