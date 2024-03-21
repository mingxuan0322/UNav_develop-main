#!/bin/bash

name=GA_01

localization_path=/media/endeleze/Endeleze_5T/WeNev/Localization
ckpt_path=$localization_path/pytorch_NetVlad/data/$name/Out/log/Sep04_06-02-25_vgg16_netvlad
image_dir=$localization_path/pytorch_NetVlad/data/$name/database
output=$localization_path/pytorch_NetVlad/data/$name
script=$localization_path/feature_extract.py


source ~/anaconda3/etc/profile.d/conda.sh

conda activate open3d

python $script --ckpt_path $ckpt_path --image_dir $image_dir --output $output
