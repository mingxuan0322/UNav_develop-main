import argparse
import torch
from pathlib import Path
import h5py
import numpy as np
import logging
from tqdm import tqdm
import pprint

from . import matchers
from .utils.base_model import dynamic_load
from .utils.parsers import names_to_pair


'''
A set of standard configurations that can be directly selected from the command
line using their name. Each is a dictionary with the following entries:
    - output: the name of the match file that will be generated.
    - model: the model configuration, as passed to a feature matcher.
'''
confs = {
    'superglue': {
        'output': 'matches-superglue',
        'model': {
            'name': 'superglue',
            'weights': 'outdoor',
            'sinkhorn_iterations': 50,
        },
    },
    'NN': {
        'output': 'matches-NN-mutual-dist.7',
        'model': {
            'name': 'nearest_neighbor',
            'mutual_check': True,
            'distance_threshold': 0.7,
        },
    }
}

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

@torch.no_grad()
def main(conf, pairs, features, export_dir, query_features=None, output_dir=None, exhaustive=False, batch_size=1):
    logging.info('Matching local features with configuration:'
                 f'\n{pprint.pformat(conf)}'
                 f'\nbatch size is {batch_size}')

    feature_path = Path(export_dir, features+'.h5')
    assert feature_path.exists(), feature_path
    feature_file = h5py.File(str(feature_path), 'r')

    if query_features is not None:
        logging.info(f'Using query_features {query_features}')
    else:
        logging.info('No query_features')
        query_features = feature_path
    assert query_features.exists(), query_features
    query_feature_file = h5py.File(str(query_features), 'r')
        
    pairs_name = pairs.stem
    if not exhaustive:
        assert pairs.exists(), pairs
        with open(pairs, 'r') as f:
            pair_list = f.read().rstrip('\n').split('\n')
    elif exhaustive:
        logging.info(f'Writing exhaustive match pairs to {pairs}.')
        assert not pairs.exists(), pairs

        # get the list of images from the feature file
        images = []
        feature_file.visititems(
            lambda name, obj: images.append(obj.parent.name.strip('/'))
            if isinstance(obj, h5py.Dataset) else None)
        images = list(set(images))

        pair_list = [' '.join((images[i], images[j]))
                     for i in range(len(images)) for j in range(i)]
        with open(str(pairs), 'w') as f:
            f.write('\n'.join(pair_list))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    Model = dynamic_load(matchers, conf['model']['name'])
    model = Model(conf['model']).eval().to(device)

    match_name = f'{features}_{conf["output"]}_{pairs_name}'
    if output_dir is None:
        output_dir = export_dir
    match_path = Path(output_dir, match_name+'.h5')
    match_path.parent.mkdir(exist_ok=True, parents=True)
    match_file = h5py.File(str(match_path), 'a')

    matched = set()
    for batch in tqdm(list(chunks(pair_list, batch_size)), smoothing=0.1):
        kplist0 = []
        kplist1 = []
        desc0=[]
        desc1=[]
        sc0=[]
        sc1=[]

        for pair in batch:
            name0, name1 = pair.split(' ')
            pair = names_to_pair(name0, name1)

            feats0, feats1 = query_feature_file[name0], feature_file[name1]
            # Avoid to recompute duplicates to save time
            if len({(name0, name1), (name1, name0)} & matched) \
               or pair in match_file:
                continue
        
            kplist0.append(feats0['keypoints'].__array__())
            kplist1.append(feats1['keypoints'].__array__())
            desc0.append(feats0['descriptors'].__array__())
            desc1.append(feats1['descriptors'].__array__())
            sc0.append(feats0['scores'].__array__())
            sc1.append(feats1['scores'].__array__())

        if len(kplist0) == 0:
            continue
        # pad feature0
        size_list=[n.shape[0] for n in kplist0]
        max_size = np.max(size_list)
        kplist0 = [ np.concatenate((n, np.zeros((max_size-n.shape[0], n.shape[1]))), axis=0) for n in kplist0]
        desc0 = [ np.concatenate((n, np.zeros((n.shape[0], max_size-n.shape[1]))), axis=1) for n in desc0]
        sc0 = [ np.concatenate((n, np.zeros((max_size-n.shape[0]))), axis=0) for n in sc0]
        # pad feature1
        size_list=[n.shape[0] for n in kplist1]
        max_size = np.max(size_list)
        kplist1 = [ np.concatenate((n, np.zeros((max_size-n.shape[0], n.shape[1]))), axis=0) for n in kplist1]
        desc1 = [ np.concatenate((n, np.zeros((n.shape[0], max_size-n.shape[1]))), axis=1) for n in desc1]
        sc1 = [ np.concatenate((n, np.zeros((max_size-n.shape[0]))), axis=0) for n in sc1]

        data = {'keypoints0':kplist0, 'descriptors0':desc0, 'scores0':sc0,'keypoints1':kplist1, 'descriptors1':desc1, 'scores1':sc1}
        data = {k: torch.from_numpy(np.array(v)).float().to(device) for k, v in data.items()}

        # some matchers might expect an image but only use its size
        data['image0'] = torch.empty((len(sc0), 1,)+tuple(feats0['image_size'])[::-1])
        data['image1'] = torch.empty((len(sc0), 1,)+tuple(feats1['image_size'])[::-1])

        pred = model(data)

        index=0
        for pair in batch:
            name0, name1 = pair.split(' ')
            pair = names_to_pair(name0, name1)

            # Avoid to recompute duplicates to save time
            if len({(name0, name1), (name1, name0)} & matched) \
               or pair in match_file:
                continue

            grp = match_file.create_group(pair)
            matches = pred['matches0'][index].cpu().short().numpy()
            grp.create_dataset('matches0', data=matches)

            if 'matching_scores0' in pred:
                scores = pred['matching_scores0'][index].cpu().half().numpy()
                grp.create_dataset('matching_scores0', data=scores)

            matched |= {(name0, name1), (name1, name0)}
            index += 1

    match_file.close()
    logging.info('Finished exporting matches.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--export_dir', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path, required=False)
    parser.add_argument('--features', type=str,
                        default='feats-superpoint-n4096-r1024')
    parser.add_argument('--query_features', type=Path, required=False)

    parser.add_argument('--pairs', type=Path, required=True)
    parser.add_argument('--conf', type=str, default='superglue',
                        choices=list(confs.keys()))
    parser.add_argument('--exhaustive', action='store_true')
    parser.add_argument('--batch', type=int, default=1)
    args = parser.parse_args()
    main(
        confs[args.conf], args.pairs, args.features,args.export_dir,
        query_features=args.query_features, output_dir=args.output_dir, batch_size=args.batch, exhaustive=args.exhaustive)
