from datetime import datetime
from pathlib import Path

from hloc import extract_features, match_features, match_features_batch, reconstruction, visualization
import argparse



parser = argparse.ArgumentParser()
parser.add_argument('--images', type=Path, required=True)
parser.add_argument('--outputs', type=Path, required=True)
parser.add_argument('--sfm_pairs', type=Path, required=False)
parser.add_argument('--use_pba', action='store_true')
parser.add_argument('--single_camera', action='store_true')
parser.add_argument('--extractor', type=str, default='superpoint_inloc', choices=['superpoint_inloc', 'superpoint_aachen', 'd2net-ss', 'hfnet_superpoint'])
parser.add_argument('--matcher', type=str, default='superglue', choices=['superglue', 'NN'])
parser.add_argument('--matcher_batch', type=int)
parser.add_argument('--min_match_score', type=float)

# skip option
parser.add_argument('--skip_extractor', action='store_true')
parser.add_argument('--skip_matcher', action='store_true')

args = parser.parse_args()

images = args.images
outputs = args.outputs

sfm_pairs = args.sfm_pairs

if sfm_pairs is None:
    sfm_pairs = images / 'pairs_netvlad50.txt'

sfm_dir = outputs / f'sfm_{args.extractor}+{args.matcher}'

feature_conf = extract_features.confs[args.extractor]
matcher_conf = match_features.confs[args.matcher]

features = feature_conf['output']
feature_file = f"{features}.h5"
match_file = f"{features}_{matcher_conf['output']}_{sfm_pairs.stem}.h5"

# ## Extract local features

print ('start at {}'.format(datetime.now()))

if not args.skip_extractor:
    extract_features.main(feature_conf, images, outputs)

print ('finished extract_features at {}'.format(datetime.now()))

# ##  Matching
#  Pairs were created using image retrieval and `hloc/pairs_from_retrieval.py`.

if not args.skip_matcher:
    if args.matcher_batch is not None:
        match_features_batch.main(matcher_conf, sfm_pairs, features, outputs, batch_size=args.matcher_batch, exhaustive=False)
    else:
        match_features.main(matcher_conf, sfm_pairs, features, outputs, exhaustive=False)
print ('finished match_features at {}'.format(datetime.now()))


# ## SfM reconstruction
# Run COLMAP on the features and matches.

# In[ ]:


reconstruction.main(
    sfm_dir,
    images,
    sfm_pairs,
    outputs / feature_file,
    outputs / match_file,
    single_camera=args.single_camera,
    min_match_score=args.min_match_score,
    use_pba=args.use_pba
)

print ('finished reconstruction at {}'.format(datetime.now()))
