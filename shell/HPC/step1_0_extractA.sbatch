#!/bin/bash

#SBATCH --job-name=2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64GB
#SBATCH --time=150:00:00
#SBATCH --gres=gpu:0
#SBATCH --output=/scratch/ay1620/UNav/log/extract.out
#SBATCH --mail-type=END
#SBATCH --mail-user=ay1620@nyu.edu
module purge

export Place=New_York_University
export Building=6_MetroTech
export Floor=Metrotech_6_Floor_4_With_Stairs
export rate=29.97

#=========================================================
base_root=/scratch/ay1620/UNav
mapping_root=$base_root/Mapping/data
src_dir=$mapping_root/src_images/$Place/$Building/$Floor/equirectangular_images
video_dir=$mapping_root/video/$Place/$Building/$Floor.mp4
maps=$mapping_root/maps/$Place/$Building/$Floor.msg
#=========================================================


overlay=/scratch/work/public/singularity/colmap-3.7-cuda10.1-cudnn7-devel-ubuntu18.04-20201207.sqf
singularity exec --nv \
            --overlay /scratch/ay1620/UNav/50G.ext3:ro \
	    --overlay $overlay:ro \
	    /scratch/ay1620/UNav/cuda10.sif \
            /bin/bash -c "
source /ext3/env.sh;
[ ! -d "$src_dir" ] && mkdir -p "$src_dir";
ffmpeg -i $video_dir -r $rate $src_dir/%05d.png;
python $mapping_root/extract_keyframe.py --maps $maps --src_dir $src_dir
"
