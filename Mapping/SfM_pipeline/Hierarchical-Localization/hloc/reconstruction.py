import argparse
import logging
from pathlib import Path
import shutil
import multiprocessing
import subprocess
import pprint

from .utils.read_write_model import read_cameras_binary
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
    # globs = ['*.jpg', '*.png', '*.jpeg', '*.JPG', '*.PNG']
    # images = []
    # for g in globs:
    #     images += list(Path(args.image_dir).glob('**/' + g))
    if len(images) == 0:
        raise IOError(f'No images found in {image_dir}.')
    # We need to create dummy features for COLMAP to import images with EXIF
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


def run_reconstruction(colmap_path, model_path, database_path, image_dir,
                       input_model=None, use_pba=False, min_num_matches=None, ):
    logging.info('Running the 3D reconstruction...')
    model_path.mkdir(exist_ok=True)

    cmd = [
        str(colmap_path), 'mapper',
        '--database_path', str(database_path),
        '--image_path', str(image_dir),
        '--output_path', str(model_path),
        '--Mapper.num_threads', str(min(multiprocessing.cpu_count(), 16))]
    if input_model:
        cmd += ['--input_path', str(input_model)]
    if min_num_matches:
        cmd += ['--Mapper.min_num_matches', str(min_num_matches)]
    if use_pba:
        cmd += ['--Mapper.ba_global_use_pba', 'true']
    logging.info(' '.join(cmd))
    ret = subprocess.call(cmd)
    if ret != 0:
        logging.warning('Problem with mapper, exiting.')
        exit(ret)

    if input_model is not None:
        largest_model = model_path
    else:
        models = list(model_path.iterdir())
        if len(models) == 0:
            logging.error('Could not reconstruct any model!')
            return False
        logging.info(f'Reconstructed {len(models)} models.')

        largest_model = None
        largest_model_num_images = 0
        for model in models:
            num_images = len(read_cameras_binary(str(model / 'cameras.bin')))
            if num_images > largest_model_num_images:
                largest_model = model
                largest_model_num_images = num_images
        assert largest_model_num_images > 0
        logging.info(f'Largest model is #{largest_model.name} '
                     'with {largest_model_num_images} images.')

    stats_raw = subprocess.check_output(
        [str(colmap_path), 'model_analyzer',
         '--path', str(largest_model)])
    stats_raw = stats_raw.decode().split("\n")
    stats = dict()
    for stat in stats_raw:
        if stat.startswith("Registered images"):
            stats['num_reg_images'] = int(stat.split()[-1])
        elif stat.startswith("Points"):
            stats['num_sparse_points'] = int(stat.split()[-1])
        elif stat.startswith("Observations"):
            stats['num_observations'] = int(stat.split()[-1])
        elif stat.startswith("Mean track length"):
            stats['mean_track_length'] = float(stat.split()[-1])
        elif stat.startswith("Mean observations per image"):
            stats['num_observations_per_image'] = float(stat.split()[-1])
        elif stat.startswith("Mean reprojection error"):
            stats['mean_reproj_error'] = float(stat.split()[-1][:-2])

    return stats

def add_images(sfm_dir, input_model, output_model, image_dir, pairs_path, features, matches,
               added_images_is_2nd_in_pairs=True,
               colmap_path='colmap', single_camera=False,
               skip_geometric_verification=False,
               use_pba=False,
               min_match_score=None, min_num_matches=None):

    assert features.exists(), features
    assert pairs_path.exists(), pairs_path
    assert matches.exists(), matches

    output_model.mkdir(exist_ok=True)

    database = sfm_dir / 'database.db'
    assert database.exists(), database

    #save a backup data base
    shutil.copy(str(database), f'{str(database)}-backup')

    with open(str(pairs_path), 'r') as f:
        image_pairs = [p.split(' ') for p in f.read().split('\n')]
    import_images(
        colmap_path, sfm_dir, image_dir, database, single_camera=single_camera, remove_features=False)
    image_ids = get_image_ids(database)

    added_image_ids = {}
    for name0, name1 in image_pairs:
        id0, id1 = image_ids[name0], image_ids[name1]
        if added_images_is_2nd_in_pairs:
            added_image_ids[name1] = id1
        else:
            added_image_ids[name0] = id0
    import_features(added_image_ids, database, features, use_replace=True)
    import_matches(image_ids, database, pairs_path, matches,
                   min_match_score, skip_geometric_verification)
    if not skip_geometric_verification:
        geometric_verification(colmap_path, database, pairs_path)
    stats = run_reconstruction(
        colmap_path, output_model, database, image_dir, input_model=input_model, use_pba=use_pba, min_num_matches=min_num_matches)
    stats['num_input_images'] = len(image_ids)
    logging.info(f'Statistics:\n{pprint.pformat(stats)}')

def main(sfm_dir, image_dir, pairs, features, matches,
         colmap_path='colmap', single_camera=False,
         skip_geometric_verification=False,
         use_pba=False,
         min_match_score=None, min_num_matches=None):

    assert features.exists(), features
    assert pairs.exists(), pairs
    assert matches.exists(), matches

    sfm_dir.mkdir(parents=True, exist_ok=True)
    database = sfm_dir / 'database.db'
    models = sfm_dir / 'models'
    models.mkdir(exist_ok=True)

    create_empty_db(database)
    import_images(
        colmap_path, sfm_dir, image_dir, database, single_camera=single_camera)
    image_ids = get_image_ids(database)
    import_features(image_ids, database, features)
    import_matches(image_ids, database, pairs, matches,
                   min_match_score, skip_geometric_verification)
    if not skip_geometric_verification:
        geometric_verification(colmap_path, database, pairs)
    stats = run_reconstruction(
        colmap_path, models, database, image_dir, use_pba=use_pba, min_num_matches=min_num_matches)
    stats['num_input_images'] = len(image_ids)
    logging.info(f'Statistics:\n{pprint.pformat(stats)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sfm_dir', type=Path, required=True)
    parser.add_argument('--image_dir', type=Path, required=True)

    parser.add_argument('--pairs', type=Path, required=True)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--matches', type=Path, required=True)

    parser.add_argument('--colmap_path', type=Path, default='colmap')

    parser.add_argument('--use_pba', action='store_true')

    parser.add_argument('--single_camera', action='store_true')
    parser.add_argument('--skip_geometric_verification', action='store_true')
    parser.add_argument('--min_match_score', type=float)
    parser.add_argument('--min_num_matches', type=int)

    parser.add_argument('--add_images', action='store_true')
    parser.add_argument('--input_model', type=Path)
    parser.add_argument('--output_model', type=Path)
    parser.add_argument('--added_images_is_2nd_in_pairs', action='store_true')

    args = parser.parse_args()

    if args.add_images:
        add_images(args.sfm_dir, args.input_model, args.output_model,
                   args.image_dir, args.pairs, args.features, args.matches,
                   added_images_is_2nd_in_pairs=args.added_images_is_2nd_in_pairs,
                   single_camera=args.single_camera,
                   skip_geometric_verification=args.skip_geometric_verification,
                   colmap_path=args.colmap_path,
                   use_pba=args.use_pba,
                   min_match_score=args.min_match_score,
                   min_num_matches=args.min_num_matches)
    else:
        main(args.sfm_dir, args.image_dir, args.pairs, args.features, args.matches,
             colmap_path=args.colmap_path, single_camera=args.single_camera,
             skip_geometric_verification=args.skip_geometric_verification,
             use_pba=args.use_pba,
             min_match_score=args.min_match_score,
             min_num_matches=args.min_num_matches)
