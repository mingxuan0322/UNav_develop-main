import argparse
import logging
from pathlib import Path
import shutil
import subprocess
import msgpack
import collections
import natsort
import numpy as np
from scipy.spatial.transform import Rotation as R
from .utils.database import COLMAPDatabase
from .triangulation import (
    import_features, import_matches, geometric_verification)

def create_empty_db(database_path):
    logging.info('Creating an empty database...')
    db = COLMAPDatabase.connect(database_path)
    db.create_tables()
    db.commit()
    db.close()

def import_images(colmap_path, sfm_dir, image_dir, database_path,
                  single_camera=False, remove_features=True):
    logging.info('Importing images into the database...')
    images = list(image_dir.iterdir())
    if len(images) == 0:
        raise IOError(f'No images found in {image_dir}.')
    dummy_dir = sfm_dir / 'dummy_features'
    dummy_dir.mkdir()
    for i in images:
        with open(str(dummy_dir / (i.name + '.txt')), 'w') as f:
            f.write('0 128')
    cmd = [
        str(colmap_path), 'feature_importer',
    '--database_path', str(database_path),
    '--image_path', str(image_dir),
    '--import_path', str(dummy_dir),
    '--ImageReader.single_camera',
    str(int(single_camera))]
    ret = subprocess.call(cmd)
    if ret != 0:
        logging.warning('Problem with feature_importer, exiting.')
        exit(ret)

    if remove_features:
        db = COLMAPDatabase.connect(database_path)
        db.execute("DELETE FROM keypoints;")
        db.execute("DELETE FROM descriptors;")
        db.commit()
        db.close()

    shutil.rmtree(str(dummy_dir))

def get_image_ids(database_path):
    db = COLMAPDatabase.connect(database_path)
    images = {}
    for name, image_id in db.execute("SELECT name, image_id FROM images;"):
        images[name] = image_id
    db.close()
    return images

def slam2world(t, r):
    r = R.from_quat(r)
    return -np.matmul(r.as_matrix().transpose(), t)

def world2colmap(t, r):
    r = R.from_quat(r)
    return -np.matmul(r.as_matrix(), t)

def create_frames(image_ids,slam_map,model_path):
    with open(slam_map, "rb") as f:
        data = msgpack.unpackb(f.read(), use_list=False, raw=False)
    keyframes = collections.OrderedDict(natsort.natsorted(data['keyframes'].items()))
    im = [i for i, v in image_ids.items()]
    dics=sorted(list(set([i.split('_')[0] for i in im])))
    yaws=max(list(set([int(i.replace('.png','').split('_')[-1]) for i in im])))+1
    ang=np.pi/yaws*2
    base={}
    for i,dic in enumerate(dics):
        key=list(keyframes.keys())[i]
        point=keyframes[key]
        trans = point["trans_cw"]
        rot = point['rot_cw']
        pos = slam2world(trans, rot)
        base.update({dic:{'rvec':rot,'tvec':pos}})
    f = open(str(model_path / 'images.txt'), 'w')
    for name,id in image_ids.items():
        yaw=-int(name.replace('.png','').split('_')[-1])*ang
        dic=name.split('_')[0]
        d = base[dic]
        rot=R.from_quat(d['rvec'])
        rvec=R.from_matrix((R.from_rotvec(np.array([0, yaw, 0])).as_matrix()) @ rot.as_matrix()).as_quat()
        tvec=world2colmap(d['tvec'],rvec)
        f.write(f'{id} {rvec[3]} {rvec[0]} {rvec[1]} {rvec[2]} {tvec[0]} {tvec[1]} {tvec[2]} 1 {name}\n\n')
    f.close()
    f = open(str(model_path / 'points3D.txt'), 'w')
    f.close()
    f = open(str(model_path / 'cameras.txt'), 'w')
    f.write(f'1 SIMPLE_RADIAL 640 360 416.747103 320.0 180.0 -0.00218346982')
    f.close()

def triangulate(database_path,image_dir,input_path,output_path,colmap_path='colmap'):
    cmd = [
        str(colmap_path), 'point_triangulator',
        '--database_path', str(database_path),
        '--image_path', str(image_dir),
        '--input_path', str(input_path),
        '--output_path', str(output_path)]
    ret = subprocess.call(cmd)
    if ret!=0:
        print('Problem with triangulation')

def main(sfm_dir, image_dir, pairs, features, matches,slam_map,
         colmap_path='colmap', single_camera=False,
         min_match_score=None):

    assert features.exists(), features
    assert pairs.exists(), pairs
    assert matches.exists(), matches

    sfm_dir.mkdir(parents=True, exist_ok=True)
    database = sfm_dir / 'database.db'
    models = sfm_dir / 'models'/'original_model'
    models.mkdir(exist_ok=True,parents=True)

    create_empty_db(database)
    import_images(
        colmap_path, sfm_dir, image_dir, database, single_camera=single_camera)
    image_ids = get_image_ids(database)
    import_features(image_ids, database, features)
    import_matches(image_ids, database, pairs, matches,
                   min_match_score,skip_geometric_verification=False)
    create_frames(image_ids, slam_map,models)
    reconstruction_path=sfm_dir / 'models'/'reconstructed_model'
    reconstruction_path.mkdir(exist_ok=True,parents=True)
    geometric_verification(colmap_path, database, pairs)
    triangulate(database,image_dir,models,reconstruction_path,colmap_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sfm_dir', type=Path, required=True)
    parser.add_argument('--image_dir', type=Path, required=True)

    parser.add_argument('--pairs', type=Path, required=True)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--matches', type=Path, required=True)
    parser.add_argument('--slam_map', type=Path, required=True)

    parser.add_argument('--colmap_path', type=Path, default='colmap')

    parser.add_argument('--single_camera', action='store_true')
    parser.add_argument('--min_match_score', type=float)
    parser.add_argument('--min_num_matches', type=int)

    parser.add_argument('--add_images', action='store_true')
    parser.add_argument('--input_model', type=Path)
    parser.add_argument('--output_model', type=Path)
    parser.add_argument('--added_images_is_2nd_in_pairs', action='store_true')

    args = parser.parse_args()

    main(args.sfm_dir, args.image_dir, args.pairs, args.features, args.matches,args.slam_map,
         colmap_path=args.colmap_path, single_camera=args.single_camera,
         min_match_score=args.min_match_score)
