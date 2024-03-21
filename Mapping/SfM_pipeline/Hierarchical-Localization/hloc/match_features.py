import argparse
import torch
from pathlib import Path
import h5py
import time
import logging
from tqdm import tqdm
import pprint
import numpy as np
from scipy.io import loadmat
from . import matchers
from .utils.base_model import dynamic_load
from .utils.parsers import names_to_pair
from sklearn.neighbors import NearestNeighbors

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

def occupy_gpu(num):
    a = torch.randn(num).to('cuda')

def get_model(conf):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    Model = dynamic_load(matchers, conf['model']['name'])
    model = Model(conf['model']).eval().to(device)
    return model

@torch.no_grad()
def do_match (name0, name1, pairs, matched, num_matches_found, model, match_file, feature_file, query_feature_file, min_match_score, min_valid_ratio):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    pair = names_to_pair(name0, name1)

    # Avoid to recompute duplicates to save time
    if len({(name0, name1), (name1, name0)} & matched) or pair in match_file:
        return num_matches_found
    data = {}
    feats0, feats1 = query_feature_file[name0], feature_file[name1]
    for k in feats1.keys():
        data[k+'0'] = feats0[k].__array__()
    for k in feats1.keys():
        data[k+'1'] = feats1[k].__array__()
    data = {k: torch.from_numpy(v)[None].float().to(device)
            for k, v in data.items()}

    # some matchers might expect an image but only use its size
    data['image0'] = torch.empty((1, 1,)+tuple(feats0['image_size'])[::-1])
    data['image1'] = torch.empty((1, 1,)+tuple(feats1['image_size'])[::-1])

    pred = model(data)
    matches = pred['matches0'][0].cpu().short().numpy()
    scores = pred['matching_scores0'][0].cpu().half().numpy()
    # if score < min_match_score, set match to invalid
    matches[ scores < min_match_score ] = -1
    num_valid = np.count_nonzero(matches > -1)
    if float(num_valid)/len(matches) > min_valid_ratio:
        v = pairs.get(name0)
        if v is None:
            v = set(())
        v.add(name1)
        pairs[name0] = v
        grp = match_file.create_group(pair)
        grp.create_dataset('matches0', data=matches)
        grp.create_dataset('matching_scores0', data=scores)
        matched |= {(name0, name1), (name1, name0)}
        num_matches_found += 1

    return num_matches_found

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def do_match_batch (name0, q_names, pairs, matched, model, match_file, feature_file, query_feature_file, min_match_score, min_valid_ratio,num_match_required,batch_size):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_matches_found = 0
    feats0=query_feature_file[name0]
    for batch in list(chunks(q_names, batch_size)):
        kplist0 = []
        kplist1 = []
        desc0 = []
        desc1 = []
        sc0 = []
        sc1 = []
        for name1 in batch:
            pair = names_to_pair(name0, name1)
            feats1 = feature_file[name1]
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
            break

        # pad feature0
        size_list = [n.shape[0] for n in kplist0]
        max_size = np.max(size_list)
        kplist0 = [np.concatenate((n, np.zeros((max_size - n.shape[0], n.shape[1]))), axis=0) for n in kplist0]
        desc0 = [np.concatenate((n, np.zeros((n.shape[0], max_size - n.shape[1]))), axis=1) for n in desc0]
        sc0 = [np.concatenate((n, np.zeros((max_size - n.shape[0]))), axis=0) for n in sc0]
        # pad feature1
        size_list = [n.shape[0] for n in kplist1]
        max_size = np.max(size_list)
        kplist1 = [np.concatenate((n, np.zeros((max_size - n.shape[0], n.shape[1]))), axis=0) for n in kplist1]
        desc1 = [np.concatenate((n, np.zeros((n.shape[0], max_size - n.shape[1]))), axis=1) for n in desc1]
        sc1 = [np.concatenate((n, np.zeros((max_size - n.shape[0]))), axis=0) for n in sc1]

        data = {'keypoints0': kplist0, 'descriptors0': desc0, 'scores0': sc0, 'keypoints1': kplist1,
                'descriptors1': desc1, 'scores1': sc1}
        data = {k: torch.from_numpy(np.array(v)).float().to(device) for k, v in data.items()}

        # some matchers might expect an image but only use its size
        data['image0'] = torch.empty((len(sc0), 1,) + tuple(feats0['image_size'])[::-1])
        data['image1'] = torch.empty((len(sc0), 1,) + tuple(feats1['image_size'])[::-1])

        pred = model(data)
        index = 0
        for name1 in batch:
            pair = names_to_pair(name0, name1)

            # Avoid to recompute duplicates to save time
            if len({(name0, name1), (name1, name0)} & matched) \
                    or pair in match_file:
                continue


            matches = pred['matches0'][index].cpu().short().numpy()
            scores = pred['matching_scores0'][index].cpu().half().numpy()
            index += 1
            matches[scores < min_match_score] = -1
            num_valid = np.count_nonzero(matches > -1)
            if float(num_valid) / len(matches) > min_valid_ratio:
                v = pairs.get(name0)
                if v is None:
                    v = set(())
                v.add(name1)
                pairs[name0] = v
                grp = match_file.create_group(pair)
                grp.create_dataset('matches0', data=matches)
                grp.create_dataset('matching_scores0', data=scores)
                matched |= {(name0, name1), (name1, name0)}
                num_matches_found += 1
                if num_matches_found>=num_match_required:
                    return num_matches_found
    return num_matches_found

@torch.no_grad()
def best_match(conf, global_feature_path, feature_path, match_output_path, equirectangular=False,yaw_seq=None,GT_path=None,GT_thre=None,query_global_feature_path=None, query_feature_path=None, num_match_required=10,
               max_try=None, min_matched=None, pair_file_path=None, num_seq=False, sample_list=None, sample_list_path=None, min_match_score=0.85, min_valid_ratio=0.09,batch_size=0):
    logging.info('Dyn Matching local features with configuration:'
                 f'\n{pprint.pformat(conf)}')


    assert global_feature_path.exists(), feature_path
    global_feature_file = h5py.File(str(global_feature_path), 'r')
    if query_global_feature_path is not None:
        logging.info(f'(Using query_global_feature_path:{query_global_feature_path}')
        query_global_feature_file = h5py.File(str(query_global_feature_path), 'r')
    else:
        query_global_feature_file = global_feature_file

    assert feature_path.exists(), feature_path
    feature_file = h5py.File(str(feature_path), 'r')
    if query_feature_path is not None:
        logging.info(f'(Using query_feature_path:{query_feature_path}')
        query_feature_file = h5py.File(str(query_feature_path), 'r')
    else:
        query_feature_file = feature_file

    match_file = h5py.File(str(match_output_path), 'a')

    if sample_list_path is not None:
        sample_list = json.load(open(str(sample_list_path, 'r')))

    # get all sample names
    if sample_list is not None:
        names = sample_list
        q_names = names
    else:
        names = []
        global_feature_file.visititems(
            lambda _, obj: names.append(obj.parent.name.strip('/'))
            if isinstance(obj, h5py.Dataset) else None)
        names = list(set(names))
        names.sort()
        q_names = []
        query_global_feature_file.visititems(
            lambda _, obj: q_names.append(obj.parent.name.strip('/'))
            if isinstance(obj, h5py.Dataset) else None)
        q_names = list(set(q_names))
        q_names.sort()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def tensor_from_names(names, hfile):
        desc = [hfile[i]['global_descriptor'].__array__() for i in names]
        if len(desc)>35000:
            desc = torch.from_numpy(np.stack(desc, 0)).float()
        else:
            desc = torch.from_numpy(np.stack(desc, 0)).to(device).float()
        return desc

    desc = tensor_from_names(names, global_feature_file)
    if query_global_feature_path is not None:
        q_desc = tensor_from_names(q_names, query_global_feature_file)
    else:
        q_desc = desc
    # descriptors are normalized, dot product indicates how close they are
    sim = torch.einsum('id,jd->ij', q_desc, desc)
    if max_try is None:
        max_try = len(names)
    topk = torch.topk(sim, max_try, dim=1).indices.cpu().numpy()

    Model = dynamic_load(matchers, conf['model']['name'])
    model = Model(conf['model']).eval().to(device)

    pairs = {}
    matched = set()

    if equirectangular:
        mat = loadmat(GT_path)
        Images = [i.replace(' ', '').replace('perspective_images/','') for i in mat['Images']]
        GT = mat['GT']
        knn = NearestNeighbors(n_jobs=1)
        knn.fit(GT)
        _, positives = knn.radius_neighbors(GT,radius=GT_thre)
        dics=[]
        for n in names:
            dics.append(n.split('_')[0])
        dics=sorted(set(dics))
        yaws=np.max([int(i.replace('.png','').split('_')[-1]) for i in names])+1
        for name0, indices in tqdm(zip(q_names, topk)):
            num_matches_found = 0
            # try sequential neighbor and yaw neighbor first
            name0_at = names.index(name0)
            p=positives[name0_at]
            neighbors_name=[Images[i] for i in p]
            n0_id = name0_at // yaws
            n0_yid = name0_at - n0_id * yaws
            names_q = []
            for s in np.arange(-num_seq, num_seq + 1):
                if s != 0:
                    index1 = n0_id + s
                    if (index1 >= 0) and (index1 < len(dics)):
                        for y in np.arange(-yaw_seq, yaw_seq + 1):
                            name=str(dics[index1])+'_'+str((n0_yid + y) % yaws).zfill(2)+'.png'
                            if name in neighbors_name:
                                names_q.append(name)
            if batch_size>0:
                names_d = [names[i] for i in indices if
                           (name0.split('_')[0] != names[i].split('_')[0]) and (names[i] not in names_q) and (
                                       names[i] in neighbors_name)]
                num_matches_found = do_match_batch(name0, names_q + names_d, pairs, matched, model, match_file,
                                                   feature_file, query_feature_file, min_match_score, min_valid_ratio,
                                                   num_match_required, batch_size)
            else:
                for name1 in names_q:
                    num_matches_found = do_match(name0, name1, pairs, matched, num_matches_found, model, match_file,
                                                 feature_file, query_feature_file, min_match_score, min_valid_ratio)

                # then the global retrievel
                for i in indices:
                    name1 = names[i]
                    if (name1 in neighbors_name) and (name1 not in names_q):
                        if query_global_feature_path is not None or name0.split('_')[0] != name1.split('_')[0]:
                            num_matches_found = do_match(name0, name1, pairs, matched, num_matches_found, model, match_file,
                                                         feature_file, query_feature_file, min_match_score, min_valid_ratio)
                            if num_matches_found >= num_match_required:
                                break
            if num_matches_found < num_match_required:
                logging.warning(
                    f'num match for {name0} found {num_matches_found} less than num_match_required:{num_match_required}')
    else:
        for name0, indices in tqdm(zip(q_names, topk)):
            num_matches_found = 0
            # try sequential neighbor first
            if num_seq is not None:
                name0_at = names.index(name0)
                begin_from = name0_at - num_seq
                if begin_from < 0:
                    begin_from = 0
                for i in range(begin_from, name0_at+num_seq):
                    if i >= len(names):
                        break
                    name1 = names[i]
                    if name0 != name1:
                        num_matches_found = do_match(name0, name1, pairs, matched, num_matches_found, model, match_file, feature_file, query_feature_file, min_match_score, min_valid_ratio)

            # then the global retrievel
            for i in indices:
                name1 = names[i]
                if query_global_feature_path is not None or name0 != name1:
                    num_matches_found = do_match(name0, name1, pairs, matched, num_matches_found, model, match_file, feature_file, query_feature_file, min_match_score, min_valid_ratio)
                    if num_matches_found >= num_match_required:
                        break

            if num_matches_found < num_match_required:
                logging.warning(f'num match for {name0} found {num_matches_found} less than num_match_required:{num_match_required}')

    match_file.close()
    if pair_file_path is not None:
        if min_matched is not None:
            pairs = {k:v for k,v in pairs.items() if len(v) >= min_matched }
        pairs_list = []
        for n0 in pairs.keys():
            for n1 in pairs.get(n0):
                pairs_list.append((n0,n1))
        with open(str(pair_file_path), 'w') as f:
            f.write('\n'.join(' '.join([i, j]) for i, j in pairs_list))
    logging.info('Finished exporting matches.')

@torch.no_grad()
def main(conf, pairs, features, export_dir, db_features=None, query_features=None, output_dir=None, exhaustive=False):
    logging.info('Matching local features with configuration:'
                 f'\n{pprint.pformat(conf)}')

    if db_features:
        feature_path = db_features
    else:
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
    for pair in tqdm(pair_list, smoothing=.1):
        name0, name1 = pair.split(' ')
        pair = names_to_pair(name0, name1)

        # Avoid to recompute duplicates to save time
        if len({(name0, name1), (name1, name0)} & matched) \
                or pair in match_file:
            continue

        data = {}
        feats0, feats1 = query_feature_file[name0], feature_file[name1]
        for k in feats1.keys():
            data[k+'0'] = feats0[k].__array__()
        for k in feats1.keys():
            data[k+'1'] = feats1[k].__array__()
        data = {k: torch.from_numpy(v)[None].float().to(device)
                for k, v in data.items()}

        # some matchers might expect an image but only use its size
        data['image0'] = torch.empty((1, 1,)+tuple(feats0['image_size'])[::-1])
        data['image1'] = torch.empty((1, 1,)+tuple(feats1['image_size'])[::-1])

        pred = model(data)
        grp = match_file.create_group(pair)
        matches = pred['matches0'][0].cpu().short().numpy()
        grp.create_dataset('matches0', data=matches)

        if 'matching_scores0' in pred:
            scores = pred['matching_scores0'][0].cpu().half().numpy()
            grp.create_dataset('matching_scores0', data=scores)

        matched |= {(name0, name1), (name1, name0)}

    match_file.close()
    logging.info('Finished exporting matches.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--export_dir', type=Path)
    parser.add_argument('--output_dir', type=Path, required=False)
    parser.add_argument('--features', type=str,
                        default='feats-superpoint-n4096-r1024')
    parser.add_argument('--db_features', type=Path)
    parser.add_argument('--query_features', type=Path, required=False)
    parser.add_argument('--occupy_gpu', type=int, default=0)

    parser.add_argument('--pairs', type=Path)
    parser.add_argument('--conf', type=str, default='superglue',
                        choices=list(confs.keys()))
    parser.add_argument('--exhaustive', action='store_true')

    # best_match
    parser.add_argument('--best_match', action='store_true')
    parser.add_argument('--batch_size', type=int, default=0)
    parser.add_argument('--global_feature_path', type=Path)
    parser.add_argument('--feature_path', type=Path)
    parser.add_argument('--query_global_feature_path', type=Path)
    parser.add_argument('--query_feature_path', type=Path)
    parser.add_argument('--match_output_path', type=Path)
    parser.add_argument('--num_match_required', type=int, default=10)
    parser.add_argument('--min_matched', type=int, default=1)
    parser.add_argument('--max_try', type=int)
    parser.add_argument('--num_seq', type=int)
    parser.add_argument('--min_match_score', type=float, default=0.85)
    parser.add_argument('--min_valid_ratio', type=float, default=0.09)
    parser.add_argument('--sample_list_path', type=Path)
    parser.add_argument('--pair_file_path', type=Path)
        #equirectangular image
    parser.add_argument('--equirectangular', action='store_true')
    parser.add_argument('--yaw_seq', type=int)
    parser.add_argument('--GT_path', type=str,default=None)
    parser.add_argument('--GT_thre', type=int, default=None)
    args = parser.parse_args()
    if args.occupy_gpu:
        occupy_gpu(args.occupy_gpu)
    if args.best_match:
        start=time.time()
        best_match(confs[args.conf], args.global_feature_path, args.feature_path, args.match_output_path,equirectangular=args.equirectangular,yaw_seq=args.yaw_seq,GT_path=args.GT_path,GT_thre=args.GT_thre,
                   query_global_feature_path=args.query_global_feature_path, query_feature_path=args.query_feature_path,
                   num_match_required=args.num_match_required, min_matched=args.min_matched, min_match_score=args.min_match_score, min_valid_ratio=args.min_valid_ratio,
                   max_try=args.max_try, num_seq=args.num_seq, sample_list_path=args.sample_list_path, pair_file_path=args.pair_file_path,batch_size=args.batch_size)
        run_time=time.time()-start
        sec=run_time%60
        mi=(run_time//60)%60
        h=run_time//3600
        print('Finish matching, running time:\t%d:%02d:%02d'%(h,mi,sec))
    else:
        main(
            confs[args.conf], args.pairs, args.features,args.export_dir,
            db_features=args.db_features, query_features=args.query_features, output_dir=args.output_dir, exhaustive=args.exhaustive)
