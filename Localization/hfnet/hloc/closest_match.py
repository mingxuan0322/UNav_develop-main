import argparse
from pathlib import Path
import h5py
import numpy as np
import torch
from tqdm import tqdm
from signal import signal, SIGINT

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

def get_model(conf):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    Model = dynamic_load(matchers, conf['model']['name'])
    model = Model(conf['model']).eval().to(device)
    return model

@torch.no_grad()
def do_match (name0, name1, pairs, matched, num_matches_found, model, match_file, feat0_file, feat1_file, min_match_score, min_valid_ratio):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if name0 != name1 :
        pair = names_to_pair(name0, name1)

        # Avoid to recompute duplicates to save time
        #if len({(name0, name1), (name1, name0)} & matched) or pair in match_file:
        if len({(name0, name1), (name1, name0)} & matched) \
                or pair in match_file:
            return num_matches_found
        data = {}
        feats0, feats1 = feat0_file[name0], feat1_file[name1]
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
        print(f'{len(matches)} matches, valid ratio {float(num_valid)/len(matches)}')
        if float(num_valid)/len(matches) > min_valid_ratio:
            pairs.append((name0, name1))
            grp = match_file.create_group(pair)
            grp.create_dataset('matches0', data=matches)
            grp.create_dataset('matching_scores0', data=scores)
            matched |= {(name0, name1), (name1, name0)}
            num_matches_found += 1
    return num_matches_found

done_match = False
def handler(signal_received, frame):
    global done_match
    # Handle any cleanup here
    print('SIGINT or CTRL-C detected. Exiting gracefully')
    done_match = True

@torch.no_grad()
def main(conf, desc1, desc2, feat1, feat2, num_matched, match_output, pair_output=None, min_match_score=0.85, min_valid_ratio=0.2):
    hfile1 = h5py.File(str(desc1), 'r')
    hfile2 = h5py.File(str(desc2), 'r')
    hfeat1 = h5py.File(str(feat1), 'r')
    hfeat2 = h5py.File(str(feat2), 'r')

    names1 = []
    hfile1.visititems(
        lambda _, obj: names1.append(obj.parent.name.strip('/'))
        if isinstance(obj, h5py.Dataset) else None)
    names1 = list(set(names1))
    names2 = []
    hfile2.visititems(
        lambda _, obj: names2.append(obj.parent.name.strip('/'))
        if isinstance(obj, h5py.Dataset) else None)
    names2 = list(set(names2))

    print (f'size of desc1:{len(names1)}, size of desc2:{len(names2)}')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def tensor_from_names(names, hfile):
        desc = [hfile[i]['global_descriptor'].__array__() for i in names]
        desc = torch.from_numpy(np.stack(desc, 0)).to(device).float()
        return desc

    db_desc1 = tensor_from_names(names1, hfile1)
    db_desc2 = tensor_from_names(names2, hfile2)
    sim = torch.einsum('id,jd->ij', db_desc1, db_desc2)
    sim = torch.reshape(sim,(-1,))
    topk = torch.topk(sim, len(names1)*len(names2)).indices.cpu().numpy()

    match_file = h5py.File(str(match_output), 'a')
    conf = confs[args.conf]
    Model = dynamic_load(matchers, conf['model']['name'])
    model = Model(conf['model']).eval().to(device)
    pairs = []
    matched = set()
    num_matches_found = 0
    #for k in tqdm(topk):
    for k in topk:
        n1 = names1[int(k/len(names2))]
        n2 = names2[k % len(names2)]
        num_matches_found = do_match(n1, n2, pairs, matched, num_matches_found, model, match_file, hfeat1, hfeat2, min_match_score, min_valid_ratio)
        print (f'num_matches_found {num_matches_found}')
        if num_matches_found >= num_matched:
            break
        if done_match:
            break

    match_file.close()
    s1=set(())
    s2=set(())
    if pair_output is not None:
        with open(str(pair_output), 'w') as f:
            f.write('\n'.join(' '.join([i, j]) for i, j in pairs))
    for i, j in pairs:
        s1.add(i)
        s2.add(j)

    if pair_output is not None:
        with open(str(pair_output.with_suffix(''))+'-in-dataset1.txt', 'w') as f:
            for s in s1:
                f.write(f'{s}\n')
        with open(str(pair_output.with_suffix(''))+'-in-dataset2.txt', 'w') as f:
            for s in s2:
                f.write(f'{s}\n')
    print('Finished exporting matches.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='superglue',
                        choices=list(confs.keys()))
    parser.add_argument('--desc1', type=Path, required=True)
    parser.add_argument('--desc2', type=Path, required=True)
    parser.add_argument('--feat1', type=Path, required=True)
    parser.add_argument('--feat2', type=Path, required=True)
    parser.add_argument('--match_output', type=Path, required=True)
    parser.add_argument('--pair_output', type=Path, required=False)
    parser.add_argument('--num_matched', type=int, required=True)
    parser.add_argument('--min_match_score', type=float, default=0.85)
    parser.add_argument('--min_valid_ratio', type=float, default=0.2)
    args = parser.parse_args()
    main(**args.__dict__)
