import argparse
import torch
from pathlib import Path
import h5py
import logging
import numpy as np
from scipy.io import loadmat
from . import matchers
from .utils.base_model import dynamic_load
from .utils.parsers import names_to_pair
from sklearn.neighbors import NearestNeighbors

def option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=0)
    parser.add_argument('--global_feature_path', type=Path)
    parser.add_argument('--feature_path', type=Path)
    parser.add_argument('--match_output_path', type=Path)
    parser.add_argument('--pair_file_path', type=Path)
    parser.add_argument('--num_match_required', type=int, default=10)
    parser.add_argument('--min_matched', type=int, default=1)
    parser.add_argument('--max_try', type=int)
    parser.add_argument('--num_seq', type=int)
    parser.add_argument('--min_match_score', type=float, default=0.85)
    parser.add_argument('--min_valid_ratio', type=float, default=0.09)
    parser.add_argument('--yaw_seq', type=int)
    parser.add_argument('--GT_path', type=str, default=None)
    parser.add_argument('--GT_thre', type=int, default=None)
    args = parser.parse_args()
    return args

def tensor_from_names(names, hfile):
    desc = [hfile[i]['global_descriptor'].__array__() for i in names]
    desc = torch.from_numpy(np.stack(desc, 0)).float()
    return desc

def find_paris(bundle,names,positives,Images,yaws,num_seq,dics,yaw_seq,max_try):
    name0,indices=bundle
    name0_at = names.index(name0)
    p = positives[name0_at]
    neighbors_name = [Images[i] for i in p]
    n0_id = name0_at // yaws
    n0_yid = name0_at - n0_id * yaws
    names_q = []
    for s in np.arange(-num_seq, num_seq + 1):
        if s != 0:
            index1 = n0_id + s
            if (index1 >= 0) and (index1 < len(dics)):
                for y in np.arange(-yaw_seq, yaw_seq + 1):
                    name = str(dics[index1]) + '_' + str((n0_yid + y) % yaws).zfill(2) + '.png'
                    if name in neighbors_name:
                        names_q.append(name)
    num=max_try-len(names_q)
    names_d=[]
    for i in indices:
        if num<=0:
            break
        if (name0.split('_')[0] != names[i].split('_')[0]) and (names[i] not in names_q) and (names[i] in neighbors_name):
            names_d.append(names[i])
            num-=1
    return names_q+names_d

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
            feats1 = feature_file[name1]
            if len({(name0, name1), (name1, name0)} & matched):
                continue
            kplist0.append(feats0['keypoints'].__array__())
            kplist1.append(feats1['keypoints'].__array__())
            desc0.append(feats0['descriptors'].__array__())
            desc1.append(feats1['descriptors'].__array__())
            sc0.append(feats0['scores'].__array__())
            sc1.append(feats1['scores'].__array__())
        if len(kplist0) == 0:
            break
        size_list = [n.shape[0] for n in kplist0]
        max_size = np.max(size_list)
        kplist0 = [np.concatenate((n, np.zeros((max_size - n.shape[0], n.shape[1]))), axis=0) for n in kplist0]
        desc0 = [np.concatenate((n, np.zeros((n.shape[0], max_size - n.shape[1]))), axis=1) for n in desc0]
        sc0 = [np.concatenate((n, np.zeros((max_size - n.shape[0]))), axis=0) for n in sc0]
        size_list = [n.shape[0] for n in kplist1]
        max_size = np.max(size_list)
        kplist1 = [np.concatenate((n, np.zeros((max_size - n.shape[0], n.shape[1]))), axis=0) for n in kplist1]
        desc1 = [np.concatenate((n, np.zeros((n.shape[0], max_size - n.shape[1]))), axis=1) for n in desc1]
        sc1 = [np.concatenate((n, np.zeros((max_size - n.shape[0]))), axis=0) for n in sc1]

        data = {'keypoints0': kplist0, 'descriptors0': desc0, 'scores0': sc0, 'keypoints1': kplist1,
                'descriptors1': desc1, 'scores1': sc1}
        data = {k: torch.from_numpy(np.array(v)).float().to(device) for k, v in data.items()}

        data['image0'] = torch.empty((len(sc0), 1,) + tuple(feats0['image_size'])[::-1])
        data['image1'] = torch.empty((len(sc0), 1,) + tuple(feats1['image_size'])[::-1])
        pred = model(data)
        index = 0
        for name1 in batch:
            pair = names_to_pair(name0, name1)
            if len({(name0, name1), (name1, name0)} & matched):
                continue
            matches = pred['matches0'][index].detach().cpu().short().numpy()
            scores = pred['matching_scores0'][index].detach().cpu().half().numpy()
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

def best_pairs(global_feature_path, feature_path,match_output_path, yaw_seq=None, GT_path=None, GT_thre=None, max_try=None,
               num_seq=False,min_match_score=0.85, min_valid_ratio=0.09,num_match_required=10,batch_size=0,pair_file_path=None):
    assert global_feature_path.exists(), feature_path
    global_feature_file = h5py.File(str(global_feature_path), 'r')
    names = []
    global_feature_file.visititems(
        lambda _, obj: names.append(obj.parent.name.strip('/'))
        if isinstance(obj, h5py.Dataset) else None)
    names = list(set(names))
    names.sort()
    desc = tensor_from_names(names, global_feature_file)
    sim = torch.einsum('id,jd->ij', desc, desc)
    topk = torch.topk(sim, len(names), dim=1).indices.numpy()
    mat = loadmat(GT_path)
    Images = [i.replace(' ', '').replace('perspective_images/', '') for i in mat['Images']]
    GT = mat['GT']
    knn = NearestNeighbors(n_jobs=1)
    knn.fit(GT)
    _, positives = knn.radius_neighbors(GT, radius=GT_thre)
    dics = []
    for n in names:
        dics.append(n.split('_')[0])
    dics = sorted(set(dics))
    yaws = np.max([int(i.replace('.png', '').split('_')[-1]) for i in names]) + 1

    device='cuda' if torch.cuda.is_available() else 'cpu'
    pairs = {}
    matched = set()
    Model = dynamic_load(matchers, 'superglue')
    model = Model({
            'name': 'superglue',
            'weights': 'outdoor',
            'sinkhorn_iterations': 50,
        }).eval().to(device)
    match_file = h5py.File(str(match_output_path), 'a')
    feature_file = h5py.File(str(feature_path), 'r')
    ind=0
    for name0, indices in zip(names, topk):
        name_pair=find_paris((name0, indices), names, positives, Images, yaws, num_seq, dics, yaw_seq,max_try)
        num_matches_found = do_match_batch(name0, name_pair, pairs, matched, model, match_file,
                                           feature_file, feature_file, min_match_score, min_valid_ratio,
                                           num_match_required, batch_size)
        if num_matches_found < num_match_required:
            logging.warning(
                f'{ind}:{len(names)} num match for {name0} found {num_matches_found} less than num_match_required:{num_match_required}')
        ind+=1

    if pair_file_path is not None:
        pairs = {k:v for k,v in pairs.items() if len(v) >= 1 }
        pairs_list = []
        for n0 in pairs.keys():
            for n1 in pairs.get(n0):
                pairs_list.append((n0,n1))
        with open(str(pair_file_path), 'w') as f:
            f.write('\n'.join(' '.join([i, j]) for i, j in pairs_list))


if __name__ == '__main__':
    args = option()
    best_pairs(args.global_feature_path, args.feature_path,args.match_output_path, yaw_seq=args.yaw_seq, GT_path=args.GT_path,
                         GT_thre=args.GT_thre, max_try=args.max_try, num_seq=args.num_seq,min_match_score=args.min_match_score, min_valid_ratio=args.min_valid_ratio,
                            num_match_required=args.num_match_required,batch_size=args.batch_size,pair_file_path=args.pair_file_path)
