import os
import cv2
import numpy as np
import argparse
from tqdm import tqdm
import threading
import scipy.io as sio
from sklearn.model_selection import train_test_split
from scipy.spatial.transform import Rotation as R
from SuperPoint_SuperGlue.base_model import dynamic_load
from SuperPoint_SuperGlue.tools import map_tensor
from SuperPoint_SuperGlue import extractors
import torch
import json
from types import SimpleNamespace
from pathlib import Path
import logging
import h5py
from multiprocessing import cpu_count
import natsort

conf = {
    'output': 'feats-superpoint-n4096-r1600',
    'model': {
        'name': 'superpoint',
        'nms_radius': 4,
        'max_keypoints': 4096,
    },
    'preprocessing': {
        'grayscale': True,
        'resize_max': 1600,
    },
}

Model = dynamic_load(extractors, conf['model']['name'])
sp_model = Model(conf['model']).eval().to('cuda')

def options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--db_src_root', default=None, required=True,
                        help='path to database root')
    parser.add_argument('--db_Topo_path', default=None, required=True,
                        help='Topometric path')
    parser.add_argument('--db_pitch_num', type=int, default=2, required=False,
                        help='number of pitch')
    parser.add_argument('--db_pitch_range', type=int, default=10, required=False,
                        help='max pitch range')
    parser.add_argument('--db_yaw_num', type=int, default=12, required=False,
                        help='number of yaw')
    parser.add_argument('--db_FOV', type=int, default=60, required=False,
                        help='FOV')
    parser.add_argument('--q_src_root', default=None, required=False,
                        help='path to database root')
    parser.add_argument('--q_Topo_path', default=None, required=False,
                        help='Topometric path')
    parser.add_argument('--q_pitch_num', type=int, default=2, required=False,
                        help='number of pitch')
    parser.add_argument('--q_pitch_range', type=int, default=10, required=False,
                        help='max pitch range')
    parser.add_argument('--q_yaw_num', type=int, default=12, required=False,
                        help='number of yaw')
    parser.add_argument('--q_FOV', type=int, default=60, required=False,
                        help='FOV')
    parser.add_argument('--frame_width', type=int, default=640, required=False,
                        help='frame width')
    parser.add_argument('--frame_height', type=int, default=360, required=False,
                        help='frame height')
    parser.add_argument('--frame_skip', type=int, default=1, required=False,
                        help='frame skip')
    parser.add_argument('--posDistThr', type=int, default=25, required=False,
                        help='posDistThr')
    parser.add_argument('--posDistSqThr', type=int, default=625, required=False,
                        help='posDistSqThr')
    parser.add_argument('--nonTrivPosDistSqThr', type=int, default=100, required=False,
                        help='nonTrivPosDistSqThr')
    parser.add_argument('--valid_ratio', type=float, default=0.1, required=False,
                        help='valid_ratio')
    parser.add_argument('--dataset', default=None, required=True,
                        help='dataset')
    parser.add_argument('--root', default=None, required=True,
                        help='path to root')
    opt = parser.parse_args()
    return opt

class Equirectangular:
    def __init__(self, img_name):
        self._img = cv2.imread(img_name, cv2.IMREAD_COLOR)
        [self._height, self._width, _] = self._img.shape

    def GetPerspective(self, FOV, THETA, PHI, height, width, RADIUS=128):
        equ_h = self._height
        equ_w = self._width
        equ_cx = (equ_w - 1) / 2.0
        equ_cy = (equ_h - 1) / 2.0
        wFOV = FOV
        hFOV = float(height) / width * wFOV
        c_x = (width - 1) / 2.0
        c_y = (height - 1) / 2.0

        wangle = (180 - wFOV) / 2.0
        w_len = 2 * RADIUS * np.sin(np.radians(wFOV / 2.0)) / np.sin(np.radians(wangle))
        w_interval = w_len / (width - 1)

        hangle = (180 - hFOV) / 2.0
        h_len = 2 * RADIUS * np.sin(np.radians(hFOV / 2.0)) / np.sin(np.radians(hangle))
        h_interval = h_len / (height - 1)
        x_map = np.zeros([height, width], np.float32) + RADIUS
        y_map = np.tile((np.arange(0, width) - c_x) * w_interval, [height, 1])
        z_map = -np.tile((np.arange(0, height) - c_y) * h_interval, [width, 1]).T
        D = np.sqrt(x_map ** 2 + y_map ** 2 + z_map ** 2)
        xyz = np.zeros([height, width, 3], float)
        xyz[:, :, 0] = (RADIUS / D * x_map)[:, :]
        xyz[:, :, 1] = (RADIUS / D * y_map)[:, :]
        xyz[:, :, 2] = (RADIUS / D * z_map)[:, :]

        y_axis = np.array([0.0, 1.0, 0.0], np.float32)
        z_axis = np.array([0.0, 0.0, 1.0], np.float32)
        [R1, _] = cv2.Rodrigues(z_axis * np.radians(THETA))
        [R2, _] = cv2.Rodrigues(np.dot(R1, y_axis) * np.radians(-PHI))

        xyz = xyz.reshape([height * width, 3]).T
        xyz = np.dot(R1, xyz)
        xyz = np.dot(R2, xyz).T
        lat = np.arcsin(xyz[:, 2] / RADIUS)
        lon = np.zeros([height * width], float)
        theta = np.arctan(xyz[:, 1] / xyz[:, 0])
        idx1 = xyz[:, 0] > 0
        idx2 = xyz[:, 1] > 0

        idx3 = ((1 - idx1) * idx2).astype(bool)
        idx4 = ((1 - idx1) * (1 - idx2)).astype(bool)

        lon[idx1] = theta[idx1]
        lon[idx3] = theta[idx3] + np.pi
        lon[idx4] = theta[idx4] - np.pi

        lon = lon.reshape([height, width]) / np.pi * 180
        lat = -lat.reshape([height, width]) / np.pi * 180
        lon = lon / 180 * equ_cx + equ_cx
        lat = lat / 90 * equ_cy + equ_cy

        persp = cv2.remap(self._img, lon.astype(np.float32), lat.astype(np.float32), cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_WRAP)

        return persp


class ImageDataset(torch.utils.data.Dataset):
    default_conf = {
        'globs': ['*.jpg', '*.png', '*.jpeg', '*.JPG', '*.PNG'],
        'grayscale': False,
        'resize_max': None,
    }

    def __init__(self, root, conf):
        self.conf = conf = SimpleNamespace(**{**self.default_conf, **conf})
        self.root = root

        self.paths = []
        for g in conf.globs:
            self.paths += list(Path(root).glob('**/' + g))
        if len(self.paths) == 0:
            raise ValueError(f'Could not find any image in root: {root}.')
        self.paths = sorted(list(set(self.paths)))
        self.paths = [i.relative_to(root) for i in self.paths]
        logging.info(f'Found {len(self.paths)} images in root {root}.')

    def __getitem__(self, idx):
        path = self.paths[idx]
        if self.conf.grayscale:
            mode = cv2.IMREAD_GRAYSCALE
        else:
            mode = cv2.IMREAD_COLOR
        image = cv2.imread(str(self.root / path), mode)
        if not self.conf.grayscale:
            image = image[:, :, ::-1]  # BGR to RGB
        if image is None:
            raise ValueError(f'Cannot read image {str(path)}.')
        image = image.astype(np.float32)
        size = image.shape[:2][::-1]
        w, h = size

        if self.conf.resize_max and max(w, h) > self.conf.resize_max:
            scale = self.conf.resize_max / max(h, w)
            h_new, w_new = int(round(h * scale)), int(round(w * scale))
            image = cv2.resize(
                image, (w_new, h_new), interpolation=cv2.INTER_LINEAR)

        if self.conf.grayscale:
            image = image[None]
        else:
            image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
        image = image / 255.

        data = {
            'name': path.as_posix(),
            'image': image,
            'original_size': np.array(size),
        }
        return data

    def __len__(self):
        return len(self.paths)


class Generate_data(threading.Thread):
    def __init__(self, db_dics, db_data, db_input_path, db_pitch_list, db_yaw_list, db_FOV,root,frame_skip, db_path, q_dics=None, q_data=None,
                 q_input_path=None, q_pitch_list=None, q_yaw_list=None, q_FOV=None):
        super(Generate_data, self).__init__()
        self.db_dics = db_dics
        self.db_data = db_data
        self.db_input_path = db_input_path
        self.db_pitch_list = db_pitch_list
        self.db_yaw_list = db_yaw_list
        self.db_FOV = db_FOV
        self.q_dics = q_dics
        self.q_data = q_data
        self.q_input_path = q_input_path
        self.q_pitch_list = q_pitch_list
        self.q_yaw_list = q_yaw_list
        self.q_FOV = q_FOV
        self.db_path = db_path
        self.root = root
        self.frame_skip = frame_skip
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.orb = cv2.ORB_create()

    def generator(self, dic, data, input_path, pitch_list, yaw_list, FOV, type):
        if (len(data)==0) or (dic.replace('.png', '') in list(data.keys())):
            equ = Equirectangular(os.path.join(input_path, dic))
            utm, Images, keypoints = [], [], {}
            Trajectory = {}
            for i, pitch in enumerate(pitch_list):
                for j, yaw in enumerate(yaw_list):
                    img = equ.GetPerspective(FOV, yaw, pitch, opt.frame_height,
                                             opt.frame_width)
                    kp = self.orb.detect(img, None)
                    if len(kp) > 100:
                        cv2.imwrite(os.path.join(self.db_path, type + dic.replace('.png', '') + '_%02d' % (
                                    i * len(pitch_list) + j) + '.png'), img)
                        Images.append(
                            os.path.join('perspective_images',
                                         type + dic.replace('.png', '') + '_%02d' % (i * len(pitch_list) + j) + '.png'))
                        try:
                            utm.append(data[dic.replace('.png', '')]['trans'])
                            Trajectory.update({type + dic.replace('.png', '') + '_%02d' % (
                                        i * len(pitch_list) + j) + '.png': {'trans': data[dic.replace('.png', '')]['trans'],
                                                                            'rot': data[dic.replace('.png', '')][
                                                                                       'rot'] - yaw / 180 * np.pi}})
                        except:
                            pass
            return {'utm': utm, 'Images': Images, 'trajectory': Trajectory}
        else:
            os.remove(os.path.join(input_path, dic))
            return None

    def extract_sp_feature(self):
        Model = dynamic_load(extractors, conf['model']['name'])
        model = Model(conf['model']).eval().to(self.device)
        loader = ImageDataset(self.db_path, conf['preprocessing'])
        loader = torch.utils.data.DataLoader(loader, num_workers=cpu_count())
        feature_path = Path(os.path.join(self.root, 'superpoints_database.h5'))
        feature_path.parent.mkdir(exist_ok=True, parents=True)
        feature_file = h5py.File(str(feature_path), 'a')

        for data in tqdm(loader):
            pred = model(map_tensor(data, lambda x: x.to(self.device)))
            pred = {k: v[0].detach().cpu().numpy() for k, v in pred.items()}

            pred['image_size'] = original_size = data['original_size'][0].numpy()
            if 'keypoints' in pred:
                size = np.array(data['image'].shape[-2:][::-1])
                scales = (original_size / size).astype(np.float32)
                pred['keypoints'] = (pred['keypoints'] + .5) * scales[None] - .5

            grp = feature_file.create_group(data['name'][0])
            for k, v in pred.items():
                grp.create_dataset(k, data=v)

            del pred

        feature_file.close()

        logging.info('Finished exporting features.')

    def netvlad_data(self, dbImage, utmDb, qImage, utmQ, type):

        numDb = len(dbImage)
        numQ = len(qImage)
        if type == 'train':
            name = 'train.mat'
        else:
            name = 'valid.mat'
        sio.savemat(os.path.join(self.root, name),
                    {'whichSet': type, 'dataset': opt.dataset, 'dbImage': dbImage,
                     'qImage': qImage, 'utmDb': utmDb, 'numDb': numDb, 'utmQ': utmQ,
                     'numQ': numQ, 'posDistThr': opt.posDistThr,
                     'posDistSqThr': opt.posDistSqThr,
                     'nonTrivPosDistSqThr': opt.nonTrivPosDistSqThr})

    def posenet_data(self, Image, pose, type):
        with open(os.path.join(self.root, 'dataset_' + type + '.txt'), 'w') as f:
            f.write('Visual Landmark Dataset V1\nImageFile, Camera Position [X Y Z W P Q R]\n\n')
            for i in range(len(Image)):
                name = Image[i].replace('perspective_images/', '')
                p = pose[i]
                f.write(f'{name} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {p[3]:.6f} {p[4]:.6f} {p[5]:.6f} {p[6]:.6f}\n')

    def colmap_data(self, Image, utm):
        sio.savemat(os.path.join(self.root, 'Colmap_GT.mat'),
                    {'Images': Image, 'GT': utm, })

    def run(self):
        db_utm = []
        pose = []
        db_Images = []
        trajectory_path = Path(os.path.join(self.root, 'trajectory.h5'))
        trajectory_path.parent.mkdir(exist_ok=True, parents=True)
        trajectory_file = h5py.File(str(trajectory_path), 'a')
        for i, dic in enumerate(tqdm(self.db_dics, desc='Training dataset:')):
            if i % self.frame_skip == 0:
                if self.q_dics:
                    d = self.generator(dic, self.db_data, self.db_input_path, self.db_pitch_list, self.db_yaw_list,
                                   self.db_FOV, 't_')
                else:
                    d = self.generator(dic, self.db_data, self.db_input_path, self.db_pitch_list, self.db_yaw_list,
                                       self.db_FOV, '')
                if d:
                    grp = trajectory_file.create_group(dic.replace('.png', ''))
                    for i in range(len(d['utm'])):
                        db_utm.append(d['utm'][i])
                        try:
                            rot = R.from_rotvec([0, 0, d['trajectory'][d['Images'][i].split('/')[-1]]['rot']]).as_quat()
                            pose.append([d['utm'][i][0], d['utm'][i][1], 0, rot[3], rot[0], rot[1], rot[2]])
                        except:
                            pass
                        db_Images.append(d['Images'][i])
                    for name in list(d['trajectory'].keys()):
                        grp1 = grp.create_group(name)
                        for k, v in d['trajectory'][name].items():
                            grp1.create_dataset(k, data=v)
                del d
        grp = trajectory_file.create_group('summary')
        grp.create_dataset('nums', data=len(self.db_dics))
        grp.create_dataset('yaws', data=len(self.db_yaw_list))
        grp.create_dataset('pitchs', data=len(self.db_pitch_list))
        trajectory_file.close()

        if self.q_dics:
            q_utm = []
            q_Images = []
            for i, dic in enumerate(tqdm(self.q_dics, desc='Query dataset:')):
                if i % self.frame_skip == 0:
                    d = self.generator(dic, self.q_data, self.q_input_path, self.q_pitch_list, self.q_yaw_list, self.q_FOV,
                                       'q_')
                    if d:
                        for i in range(len(d['utm'])):
                            q_utm.append(d['utm'][i])
                            q_Images.append(d['Images'][i])
                    del d
            q_utm = np.array(q_utm)
            q_tImage, q_vImage, q_tutm, q_vutm = train_test_split(q_Images, q_utm, test_size=opt.valid_ratio)
            db_tImage, db_vImage, db_tutm, db_vutm = train_test_split(db_Images, db_utm, test_size=opt.valid_ratio)
            self.netvlad_data(db_tImage, db_tutm, q_tImage, q_tutm, 'train')
            self.netvlad_data(db_vImage, db_vutm, q_vImage, q_vutm, 'val')
        else:
            db_utm = np.array(db_utm)
            self.colmap_data(db_Images, db_utm)

        # tImage, vImage, tpose, vpose = train_test_split(db_Images, pose, test_size=opt.valid_ratio)
        # self.posenet_data(tImage, tpose, 'train')
        # self.posenet_data(vImage, vpose, 'test')
        # self.extract_sp_feature()


def main(opt):
    db_input_path = opt.db_src_root
    print(db_input_path)
    db_dics = sorted(os.listdir(db_input_path))
    db_data=[]
    if os.path.exists(opt.db_Topo_path):
        with open(opt.db_Topo_path, 'r') as f:
            db_data = json.load(f)['keyframes']
    db_yaw_list = np.linspace(0, 360, opt.db_yaw_num + 1)[:-1]
    db_pitch_list = np.linspace(-opt.db_pitch_range / 2, opt.db_pitch_range / 2, opt.db_pitch_num)

    root = opt.root
    db_path = os.path.join(root, 'perspective_images')
    # q_path=os.path.join(root,'query')
    if not os.path.exists(db_path):
        os.makedirs(db_path)

    if opt.q_src_root:
        q_input_path = opt.q_src_root
        q_dics = sorted(os.listdir(q_input_path))
        with open(opt.q_Topo_path, 'r') as f:
            q_data = json.load(f)['keyframes']
        q_yaw_list = np.linspace(0, 360, opt.q_yaw_num + 1)[:-1]
        q_pitch_list = np.linspace(-opt.q_pitch_range / 2, opt.q_pitch_range / 2, opt.q_pitch_num)
        generator = Generate_data(db_dics, db_data, db_input_path, db_pitch_list, db_yaw_list, opt.db_FOV,root, opt.frame_skip, db_path , q_dics, q_data,
                                  q_input_path, q_pitch_list, q_yaw_list, opt.q_FOV)
    else:
        generator = Generate_data(db_dics, db_data, db_input_path, db_pitch_list, db_yaw_list, opt.db_FOV, root, opt.frame_skip, db_path)
    generator.start()
    generator.done = True
    generator.join()


if __name__ == '__main__':
    opt = options()
    main(opt)
