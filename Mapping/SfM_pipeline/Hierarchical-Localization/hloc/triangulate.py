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
from .triangulation import geometric_verification

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
