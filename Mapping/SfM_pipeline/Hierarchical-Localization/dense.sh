#!/bin/bash
usage() {
    echo "Usage:"
    echo " dense.sh [ -i image_dir ] [ -s sparse_model ] [ -d dense_workspace ]"
}
image_path=""
sparse_model=""
dense_workspace=""

while getopts "i:s:d:" options; do
    case "${options}" in
	i)
	    image_path=${OPTARG}
	    ;;
	s)
	    sparse_model=${OPTARG}
	    ;;
	d)
	    dense_workspace=${OPTARG}
	    ;;
        *)
            echo "error: unrecognized option ($arg)" 1>&2
            usage
            exit 1
    esac
done
if [ -z "$dense_workspace" ]; then
    dense_workspace=${sparse_model}/../dense
fi

begin_sec=`date +%s`
echo "Start at $(date)"

colmap image_undistorter --image_path $image_path --input_path $sparse_model --output_path $dense_workspace --output_type COLMAP --max_image_size 2000
colmap patch_match_stereo --workspace_path $dense_workspace --workspace_format COLMAP --PatchMatchStereo.geom_consistency true
colmap stereo_fusion --workspace_path $dense_workspace --workspace_format COLMAP --input_type geometric --output_path $dense_workspace/fused.ply
#colmap poisson_mesher --input_path $dense_workspace/fused.ply --output_path $dense_workspace/meshed-poisson.ply
#colmap delaunay_mesher --input_path $dense_workspace --output_path $dense_workspace/meshed-delaunay.ply

end_sec1=`date +%s`
wall_clock=$(($end_sec1-$begin_sec))
echo "Finish at $(date), wall clock $wall_clock (sec)"
