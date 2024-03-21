#!/bin/bash

#SBATCH --job-name=2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64GB
#SBATCH --time=150:00:00
#SBATCH --gres=gpu:0
#SBATCH --output=/scratch/ay1620/WeNav/log/e2p.out
#SBATCH --mail-type=END
#SBATCH --mail-user=ay1620@nyu.edu
module purge

#=========================================================
base_root=/scratch/ay1620/WeNav
mapping_root=$base_root/Mapping
topo_root=$mapping_root/Topomap
topo_out=$topo_root/Output/$video
#=========================================================
data_path=$mapping_root/data/src_images/$video
equirect_image_path=$data_path/equirectangular_images
#=========================================================
db_Topo_path=$topo_out/slam_data.json
db_pitch_num=1
db_pitch_range=20
db_yaw_num=18
db_FOV=75
frame_width=640
frame_height=360
frame_skip=1

overlay=/scratch/work/public/singularity/colmap-3.7-cuda10.1-cudnn7-devel-ubuntu18.04-20201207.sqf
singularity exec --nv \
            --overlay /scratch/ay1620/WeNav/50G.ext3:ro \
	    --overlay $overlay:ro \
	    /scratch/ay1620/WeNav/cuda10.sif \
            /bin/bash -c "
source /ext3/env.sh;
python /scratch/ay1620/WeNav/Localization/Equirec2Perspec.py --root $data_path --db_src_root $equirect_image_path --db_Topo_path $db_Topo_path \
--db_pitch_num $db_pitch_num --db_yaw_num $db_yaw_num --db_pitch_range $db_pitch_range --db_FOV $db_FOV \
--frame_width $frame_width --frame_height $frame_height --frame_skip $frame_skip \
--dataset $video"
