import argparse
import logging
from pathlib import Path
import h5py
import numpy as np
import torch

from .utils.parsers import parse_image_lists_with_intrinsics


def main(descriptors, output, num_matched, query_descriptors=None,
         db_prefix=None, query_prefix=None, db_list=None, query_list=None):
    logging.info('Extracting image pairs from a retrieval database.')
    has_query_descriptors = True
    if query_descriptors is not None:
        logging.info(f'query_descriptors is {query_descriptors}')
    else:
        logging.info('no query_descriptors')
        query_descriptors = descriptors
        has_query_descriptors = False
        
    hfile = h5py.File(str(descriptors), 'r')
    q_hfile = h5py.File(str(query_descriptors), 'r')

    if db_prefix and query_prefix:
        if db_prefix == 'ALL':
            db_prefix = ''
        if query_prefix == 'ALL':
            query_prefix = ''
        names = []
        hfile.visititems(
            lambda _, obj: names.append(obj.parent.name.strip('/'))
            if isinstance(obj, h5py.Dataset) else None)
        names = list(set(names))
        db_names = [n for n in names if n.startswith(db_prefix)]
        q_names = []
        q_hfile.visititems(
            lambda _, obj: q_names.append(obj.parent.name.strip('/'))
            if isinstance(obj, h5py.Dataset) else None)
        q_names = list(set(q_names))
        query_names = [n for n in q_names if n.startswith(query_prefix)]
        assert len(db_names)
        assert len(query_names)
    elif db_list and query_list:
        db_names = [
            n for n, _ in parse_image_lists_with_intrinsics(db_list)]
        query_names = [
            n for n, _ in parse_image_lists_with_intrinsics(query_list)]
    else:
        raise ValueError('Provide either prefixes of DB and query names, '
                         'or paths to lists of DB and query images.')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def tensor_from_names(names, hfile):
        desc = [hfile[i]['global_descriptor'].__array__() for i in names]
        desc = torch.from_numpy(np.stack(desc, 0)).to(device).float()
        return desc

    db_desc = tensor_from_names(db_names, hfile)
    query_desc = tensor_from_names(query_names, q_hfile)
    sim = torch.einsum('id,jd->ij', query_desc, db_desc)
    topk = torch.topk(sim, num_matched, dim=1).indices.cpu().numpy()

    pairs = []
    for query, indices in zip(query_names, topk):
        for i in indices:
            if has_query_descriptors or query != db_names[i] :
                pair = (query, db_names[i])
                pairs.append(pair)

    logging.info(f'Found {len(pairs)} pairs.')
    with open(output, 'w') as f:
        f.write('\n'.join(' '.join([i, j]) for i, j in pairs))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--descriptors', type=Path, required=True)
    parser.add_argument('--query_descriptors', type=Path, required=False)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--num_matched', type=int, required=True)
    parser.add_argument('--db_prefix', type=str)
    parser.add_argument('--query_prefix', type=str)
    parser.add_argument('--db_list', type=Path)
    parser.add_argument('--query_list', type=Path)
    args = parser.parse_args()
    main(**args.__dict__)
