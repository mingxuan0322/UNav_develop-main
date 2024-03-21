import argparse
import logging
import numpy as np
from pathlib import Path
from collections import defaultdict
import h5py
from tqdm import tqdm
import pickle
import pycolmap

from .utils.read_write_model import read_model
from .utils.parsers import (
    parse_image_lists_with_intrinsics, parse_retrieval, names_to_pair)


def do_covisibility_clustering(frame_ids, all_images, points3D):
    clusters = []
    visited = set()

    for frame_id in frame_ids:
        # Check if already labeled
        if frame_id in visited:
            continue

        # New component
        clusters.append([])
        queue = {frame_id}
        while len(queue):
            exploration_frame = queue.pop()

            # Already part of the component
            if exploration_frame in visited:
                continue
            visited.add(exploration_frame)
            clusters[-1].append(exploration_frame)

            observed = all_images[exploration_frame].point3D_ids
            connected_frames = set(
                j for i in observed if i != -1 for j in points3D[i].image_ids)
            connected_frames &= set(frame_ids)
            connected_frames -= visited
            queue |= connected_frames

    clusters = sorted(clusters, key=len, reverse=True)
    return clusters


def pose_from_cluster_with_feature(query_feature, qinfo, db_ids, db_images, points3D, feature_match, thresh):
    kpq = query_feature['keypoints'].__array__()
    kp_idx_to_3D = defaultdict(list)
    kp_idx_to_3D_to_db = defaultdict(lambda: defaultdict(list))
    num_matches = 0

    for i, db_id in enumerate(db_ids):
        db_name = db_images[db_id].name
        points3D_ids = db_images[db_id].point3D_ids

        matches = feature_match.get(db_name)
        if matches is not None:
            matches = matches['matches0'].__array__()
            valid = np.where(matches > -1)[0]
            valid = valid[points3D_ids[matches[valid]] != -1]
            num_matches += len(valid)

            for idx in valid:
                id_3D = points3D_ids[matches[idx]]
                kp_idx_to_3D_to_db[idx][id_3D].append(i)
                # avoid duplicate observations
                if id_3D not in kp_idx_to_3D[idx]:
                    kp_idx_to_3D[idx].append(id_3D)

    idxs = list(kp_idx_to_3D.keys())
    mkp_idxs = [i for i in idxs for _ in kp_idx_to_3D[i]]
    mkpq = kpq[mkp_idxs]
    mkpq += 0.5  # COLMAP coordinates

    mp3d_ids = [j for i in idxs for j in kp_idx_to_3D[i]]
    mp3d = [points3D[j].xyz for j in mp3d_ids]
    mp3d = np.array(mp3d).reshape(-1, 3)

    # mostly for logging and post-processing
    mkp_to_3D_to_db = [(j, kp_idx_to_3D_to_db[i][j])
                       for i in idxs for j in kp_idx_to_3D[i]]

    camera_model, width, height, params = qinfo
    cfg = {
        'model': camera_model,
        'width': width,
        'height': height,
        'params': params,
    }
    ret = pycolmap.absolute_pose_estimation(mkpq, mp3d, cfg, thresh)
    ret['cfg'] = cfg
    ret['num_matches'] = num_matches
    return ret, mkpq, mp3d, mp3d_ids, num_matches, (mkp_idxs, mkp_to_3D_to_db)

def pose_from_cluster(qname, qinfo, db_ids, db_images, points3D,
                      feature_file, match_file, thresh):
    kpq = feature_file[qname]['keypoints'].__array__()
    kp_idx_to_3D = defaultdict(list)
    kp_idx_to_3D_to_db = defaultdict(lambda: defaultdict(list))
    num_matches = 0

    for i, db_id in enumerate(db_ids):
        db_name = db_images[db_id].name
        points3D_ids = db_images[db_id].point3D_ids

        pair = names_to_pair(qname, db_name)
        matches = match_file.get(pair)
        if matches is not None:
            matches = matches['matches0'].__array__()
            valid = np.where(matches > -1)[0]
            valid = valid[points3D_ids[matches[valid]] != -1]
            num_matches += len(valid)

            for idx in valid:
                id_3D = points3D_ids[matches[idx]]
                kp_idx_to_3D_to_db[idx][id_3D].append(i)
                # avoid duplicate observations
                if id_3D not in kp_idx_to_3D[idx]:
                    kp_idx_to_3D[idx].append(id_3D)

    idxs = list(kp_idx_to_3D.keys())
    mkp_idxs = [i for i in idxs for _ in kp_idx_to_3D[i]]
    mkpq = kpq[mkp_idxs]
    mkpq += 0.5  # COLMAP coordinates

    mp3d_ids = [j for i in idxs for j in kp_idx_to_3D[i]]
    mp3d = [points3D[j].xyz for j in mp3d_ids]
    mp3d = np.array(mp3d).reshape(-1, 3)

    # mostly for logging and post-processing
    mkp_to_3D_to_db = [(j, kp_idx_to_3D_to_db[i][j])
                       for i in idxs for j in kp_idx_to_3D[i]]

    camera_model, width, height, params = qinfo
    cfg = {
        'model': camera_model,
        'width': width,
        'height': height,
        'params': params,
    }
    ret = pycolmap.absolute_pose_estimation(mkpq, mp3d, cfg, thresh)
    ret['cfg'] = cfg
    return ret, mkpq, mp3d, mp3d_ids, num_matches, (mkp_idxs, mkp_to_3D_to_db)

# a class can be used to repeatly localize frames,
class SfMLocalizer(object):
    def __init__(self, reference_sfm, ransac_thresh=12, covisibility_clustering=False):
        logging.info('Reading 3D model...')
        _, self.db_images, self.points3D = read_model(str(reference_sfm), '.bin')
        self.db_name_to_id = {image.name: i for i, image in self.db_images.items()}
        self.ransac_thresh = ransac_thresh
        self.covisibility_clustering = covisibility_clustering

    def localize(self, query_feature, db_matched_names, matched_features, camera_intrinsics):
        # parse camera_intrinsics
        data = camera_intrinsics.split(' ')
        camera_model, width, height = data[:3]
        params = np.array(data[3:], float)
        camera_info = (camera_model, int(width), int(height), params)

        db_names = db_matched_names
        db_ids = []
        for n in db_names:
            if n not in self.db_name_to_id:
                logging.warning(f'Image {n} was retrieved but not in database')
                continue
            db_ids.append(self.db_name_to_id[n])

        ret = {'success':False}
        if self.covisibility_clustering:
            clusters = do_covisibility_clustering(db_ids, self.db_images, self.points3D)
            best_inliers = 0
            best_cluster = None
            logs_clusters = []
            for i, cluster_ids in enumerate(clusters):
                ret, mkpq, mp3d, mp3d_ids, num_matches, _ = pose_from_cluster_with_feature(
                    query_feature, camera_info, cluster_ids, self.db_images, self.points3D,
                    matched_features, thresh=self.ransac_thresh)
                if ret['success'] and ret['num_inliers'] > best_inliers:
                    best_cluster = i
                    best_inliers = ret['num_inliers']
                logs_clusters.append({
                    'cluster_ids': cluster_ids,
                    'PnP_ret': ret,
                    'keypoints_query': mkpq,
                    'points3D_xyz': mp3d,
                    'points3D_ids': mp3d_ids,
                    'num_matches': num_matches,
                })
            # logging.info(f'# inliers: {best_inliers}')
            if best_cluster is not None:
                ret = logs_clusters[best_cluster]['PnP_ret']
        else:
            ret, mkpq, mp3d, mp3d_ids, num_matches, map_ = pose_from_cluster_with_feature(
                query_feature, camera_info, db_ids, self.db_images, self.points3D,
                matched_features, thresh=self.ransac_thresh)
            # logging.info(f'# inliers: {ret["num_inliers"]}')

        if not ret['success']:
            if len(db_ids) > 0:
                closest = self.db_images.get(db_ids[0])
                if closest is not None:
                    ret['closest'] = {'qvec':closest.qvec, 'tvec':closest.tvec}

        return ret

def main(reference_sfm, queries, retrieval, features, matches, results,
         ransac_thresh=12, covisibility_clustering=False):

    assert reference_sfm.exists(), reference_sfm
    assert retrieval.exists(), retrieval
    assert features.exists(), features
    assert matches.exists(), matches

    queries = parse_image_lists_with_intrinsics(queries)
    retrieval_dict = parse_retrieval(retrieval)

    logging.info('Reading 3D model...')
    _, db_images, points3D = read_model(str(reference_sfm), '.bin')
    db_name_to_id = {image.name: i for i, image in db_images.items()}

    feature_file = h5py.File(features, 'r')
    match_file = h5py.File(matches, 'r')

    poses = {}
    logs = {
        'features': features,
        'matches': matches,
        'retrieval': retrieval,
        'loc': {},
    }
    logging.info('Starting localization...')
    for qname, qinfo in tqdm(queries):
        db_names = retrieval_dict[qname]
        db_ids = []
        for n in db_names:
            if n not in db_name_to_id:
                logging.warning(f'Image {n} was retrieved but not in database')
                continue
            db_ids.append(db_name_to_id[n])

        if covisibility_clustering:
            clusters = do_covisibility_clustering(db_ids, db_images, points3D)
            best_inliers = 0
            best_cluster = None
            logs_clusters = []
            for i, cluster_ids in enumerate(clusters):
                ret, mkpq, mp3d, mp3d_ids, num_matches, _ = pose_from_cluster(
                    qname, qinfo, cluster_ids, db_images, points3D,
                    feature_file, match_file, thresh=ransac_thresh)
                if ret['success'] and ret['num_inliers'] > best_inliers:
                    best_cluster = i
                    best_inliers = ret['num_inliers']
                logs_clusters.append({
                    'cluster_ids': cluster_ids,
                    'PnP_ret': ret,
                    'keypoints_query': mkpq,
                    'points3D_xyz': mp3d,
                    'points3D_ids': mp3d_ids,
                    'num_matches': num_matches,
                })
            # logging.info(f'# inliers: {best_inliers}')
            if best_cluster is not None:
                ret = logs_clusters[best_cluster]['PnP_ret']
                poses[qname] = (ret['qvec'], ret['tvec'])
            logs['loc'][qname] = {
                'db_ids': db_ids,
                'best_cluster': best_cluster,
                'log_clusters': logs_clusters,
            }
        else:
            ret, mkpq, mp3d, mp3d_ids, num_matches, map_ = pose_from_cluster(
                qname, qinfo, db_ids, db_images, points3D,
                feature_file, match_file, thresh=ransac_thresh)
            # logging.info(f'# inliers: {ret["num_inliers"]}')

            if ret['success']:
                poses[qname] = (ret['qvec'], ret['tvec'])
            else:
                if len(db_ids) > 0:
                    closest = db_images.get(db_ids[0])
                    if closest is not None:
                        poses[qname] = (closest.qvec, closest.tvec)
            logs['loc'][qname] = {
                'db': db_ids,
                'PnP_ret': ret,
                'keypoints_query': mkpq,
                'points3D_xyz': mp3d,
                'points3D_ids': mp3d_ids,
                'num_matches': num_matches,
                'keypoint_index_to_db': map_,
            }

    logging.info(f'Localized {len(poses)} / {len(queries)} images.')
    logging.info(f'Writing poses to {results}...')
    with open(results, 'w') as f:
        cnt = 1
        for q in poses:
            qvec, tvec = poses[q]
            qvec = ' '.join(map(str, qvec))
            tvec = ' '.join(map(str, tvec))
            name = q.split('/')[-1]
            # make it look like colmap model text format
            f.write(f'{cnt} {qvec} {tvec} 100 {name}\n20 30 -1\n')
            cnt += 1

    logs_path = f'{results}_logs.pkl'
    logging.info(f'Writing logs to {logs_path}...')
    with open(logs_path, 'wb') as f:
        pickle.dump(logs, f)
    logging.info('Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference_sfm', type=Path, required=True)
    parser.add_argument('--queries', type=Path, required=True)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--matches', type=Path, required=True)
    parser.add_argument('--retrieval', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--ransac_thresh', type=float, default=12.0)
    parser.add_argument('--covisibility_clustering', action='store_true')
    args = parser.parse_args()
    main(**args.__dict__)
