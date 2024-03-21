#!/bin/bash

#SBATCH --job-name=4_0
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64GB
#SBATCH --time=150:00:00
#SBATCH --gres=gpu:0
#SBATCH --output=/scratch/ay1620/WeNav/log/s2c.out
#SBATCH --mail-type=END
#SBATCH --mail-user=ay1620@nyu.edu
module purge

#=========================================================
base_root=/scratch/ay1620/WeNav
mapping_root=$base_root/Mapping
topo_root=$mapping_root/Topomap
topo_out=$topo_root/Output/$video
work_path=$mapping_root/data/maps
#=========================================================
sfm_map=$work_path/$video
output_model=$sfm_map/models
reconstructed_model=$output_model/reconstructed_model
#=========================================================
radius=1000
min_distance=5
#=========================================================

overlay=/scratch/work/public/singularity/colmap-3.7-cuda10.1-cudnn7-devel-ubuntu18.04-20201207.sqf
singularity exec --nv \
            --overlay /scratch/ay1620/WeNav/50G.ext3:ro \
	    --overlay $overlay:ro \
	    /scratch/ay1620/WeNav/cuda10.sif \
            /bin/bash -c "
source /ext3/env.sh;
python /scratch/ay1620/WeNav/Mapping/Topomap/colmap2topometric.py --outf $topo_out --maps $reconstructed_model;
python /scratch/ay1620/WeNav/Planner/Path_finder.py --topomap_path $topo_out --radius $radius --min_distance $min_distance"
