import os
import cv2
import numpy as np
import argparse
from tqdm import tqdm
import threading
import scipy.io as sio
from sklearn.model_selection import train_test_split
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

Model =dynamic_load(extractors, conf['model']['name'])
sp_model = Model(conf['model']).eval().to('cuda')

def options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_root', default=None, required=True,
                        help='path to database root')
    parser.add_argument('--Topo_path', default=None, required=True,
                        help='Topometric path')
    parser.add_argument('--pitch_num', type=int,default=2, required=False,
                        help='number of pitch')
    parser.add_argument('--pitch_range',type=int, default=10, required=False,
                        help='max pitch range')
    parser.add_argument('--yaw_num', type=int, default=12, required=False,
                        help='number of yaw')
    parser.add_argument('--frame_width', type=int, default=640, required=False,
                        help='frame width')
    parser.add_argument('--frame_height', type=int, default=360, required=False,
                        help='frame height')
    parser.add_argument('--FOV', type=int, default=60, required=False,
                        help='FOV')
    parser.add_argument('--posDistThr', type=int, default=25, required=False,
                        help='posDistThr')
    parser.add_argument('--posDistSqThr', type=int, default=625, required=False,
                        help='posDistSqThr')
    parser.add_argument('--nonTrivPosDistSqThr', type=int, default=100, required=False,
                        help='nonTrivPosDistSqThr')
    parser.add_argument('--valid_ratio', type=float, default=0.1, required=True,
                        help='valid_ratio')
    parser.add_argument('--dataset', default=None, required=True,
                        help='dataset')
    parser.add_argument('--query_ratio', type=float, default=0.33, required=True,
                        help='dataset')
    parser.add_argument('--root', default=None, required=True,
                        help='path to root')
    opt = parser.parse_args()
    return opt

class Equirectangular:
    def __init__(self, img_name, kpts, lm_id):
        self._img = cv2.imread(img_name, cv2.IMREAD_COLOR)
        [self._height, self._width, _] = self._img.shape
        # self.kpts = kpts
        # self.lm_id = lm_id
        # self.lm_map = self.get_lm_map()

    # def get_lm_map(self):
    #     x, y, _ = self._img.shape
    #     im = np.ones((x, y,2), dtype=np.uint32) * -1
    #     for i, kp in enumerate(self.kpts):
    #         im[int(kp[1]), int(kp[0]),0] = i
    #         im[int(kp[1]), int(kp[0]), 1] = self.lm_id[i]
    #     return im
    #
    # def undistort_kpts(self, lon, lat):
    #     x, y = lon.shape
    #     undist_kpts = []
    #     lm = []
    #     for i in range(x):
    #         for j in range(y):
    #             xx, yy = int(lon[i, j]), int(lat[i, j])
    #             for shre_x in [-1,0,1]:
    #                 for shre_y in [-1,0,1]:
    #                     if (yy-shre_y>0) and (yy-shre_y<self._height) and (xx-shre_x>0) and (xx-shre_x<self._width):
    #                         id_ = self.lm_map[yy-shre_y, xx-shre_x,1]
    #                         if id_ != -1:
    #                             undist_kpts.append([i, j])
    #                             lm.append(id_)
    #     return undist_kpts, lm

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

        #kpts, lm= self.undistort_kpts(lon, lat)
        kpts,lm=None,None

        return persp,kpts, lm

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
            self.paths += list(Path(root).glob('**/'+g))
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
            h_new, w_new = int(round(h*scale)), int(round(w*scale))
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
    def __init__(self, dics,data, db_path, input_path, pitch_list, yaw_list,root):
        super(Generate_data, self).__init__()
        self.dics=dics
        self.data=data
        self.db_path=db_path
        self.input_path=input_path
        self.pitch_list=pitch_list
        self.yaw_list=yaw_list
        self.root=root
        self.device='cuda' if torch.cuda.is_available() else 'cpu'

    def generator(self,dic):
        if dic.replace('.png', '') in list(self.data.keys()):
            des_folder = os.path.join(self.db_path, dic.replace('.png', ''))
            if not os.path.exists(des_folder):
                os.mkdir(des_folder)
            else:
                return None
            # kpts = [i['pt'] for i in self.data[dic.replace('.png', '')]['keypts']]
            # lm_id = self.data[dic.replace('.png', '')]['lm_ids']
            kpts,lm_id=None,None
            equ = Equirectangular(os.path.join(self.input_path, dic), kpts, lm_id)
            utm, Images, keypoints = [], [], {}
            for i, pitch in enumerate(self.pitch_list):
                for j, yaw in enumerate(self.yaw_list):
                    img, kp, lm = equ.GetPerspective(opt.FOV, yaw, pitch, opt.frame_height,
                                                     opt.frame_width)

                    cv2.imwrite(os.path.join(des_folder, 'pitch_%d_yaw_%d' % (i, j) + '.png'), img)

                    utm.append(self.data[dic.replace('.png', '')]['trans'])
                    Images.append(
                        os.path.join('database', dic.replace('.png', ''), 'pitch_%d_yaw_%d' % (i, j) + '.png'))
                    #keypoints.update({'pitch_%d_yaw_%d.png' % (i, j):{'kpts': kp, 'lm': lm,'rot':self.data[dic.replace('.png', '')]['rot']-yaw/180*np.pi}})
            return {'utm': utm, 'Images': Images}#, 'kpts': keypoints}
        else:
            os.remove(os.path.join(self.input_path, dic))
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

    def get_useful_feature(self):
        images = []
        orb_file = h5py.File(os.path.join(self.root,'orb_feature_landmark.h5'), 'r')
        sp_file = h5py.File(os.path.join(self.root,'superpoints_database.h5'), 'r')
        orb_file.visititems(
            lambda name, obj: images.append(obj.parent.name.strip('/'))
            if isinstance(obj, h5py.Dataset) else None)
        images = natsort.natsorted(list(set(images)))

        def read_file(file):
            l = {}
            for k, v in file.items():
                l.update({k: v.__array__()})
            return l

        useful_sp_path = Path(os.path.join(self.root,'orb_feature_landmark.h5'))
        useful_sp_path.parent.mkdir(exist_ok=True, parents=True)
        useful_sp_file = h5py.File(str(useful_sp_path), 'a')
        for im in tqdm(images):
            feats1 = orb_file[im]
            feats2 = sp_file[im]
            l1 = read_file(feats1)
            l2 = read_file(feats2)
            kp1, lm, kp2, desc2, size2, scores2 = l1['kpts'], l1['lm'], l2['keypoints'], l2['descriptors'], l2[
                'image_size'], l2['scores']
            kp_, desc_, lm_, scores_ = [], [], [], []
            for j, pt in enumerate(kp1):
                for i, pt1 in enumerate(kp2):
                    if ((pt1[0] - pt[0]) ** 2 + (pt1[1] - pt[1]) ** 2) ** (0.5) < 5:
                        kp_.append([pt1[0], pt1[1]])
                        desc_.append(desc2[:, i].reshape(-1, 1))
                        scores_.append(scores2[i])
                        lm_.append(int(lm[j]))
            if len(kp_) > 0:
                kp_, desc_, lm_, scores_ = np.array(kp_), torch.from_numpy(
                    np.concatenate(([c for c in desc_]), 1)), np.array(lm_), np.array(scores_)
                pred = {'descriptors': desc_, 'image_size': size2, 'keypoints': kp_, 'scores': scores_, 'lm': lm_}
                grp = useful_sp_file.create_group(im)
                for k, v in pred.items():
                    grp.create_dataset(k, data=v)
                del pred
        useful_sp_file.close()

    def run(self):
        utm = []
        Images = []
        # orb_path = Path(os.path.join(self.root, 'orb_feature_landmark.h5'))
        # orb_path.parent.mkdir(exist_ok=True, parents=True)
        # orb_file = h5py.File(str(orb_path), 'a')
        for dic in tqdm(self.dics[:80]):
            d =self.generator(dic)
            if d:
                # grp = orb_file.create_group(dic.replace('.png', ''))
                for i in range(len(d['utm'])):
                    utm.append(d['utm'][i])
                    Images.append(d['Images'][i])
                # for name in list(d['kpts'].keys()):
                #     grp1=grp.create_group(name)
                #     for k, v in d['kpts'][name].items():
                #         grp1.create_dataset(k, data=v)
            del d
        # grp = orb_file.create_group('summary')
        # grp.create_dataset('nums', data=len(self.dics))
        # grp.create_dataset('yaws', data=len(self.yaw_list))
        # grp.create_dataset('pitchs', data=len(self.pitch_list))
        # orb_file.close()

        utm = np.array(utm)
        tImage, vImage, tutm, vutm = train_test_split(Images, utm, test_size=opt.valid_ratio)
        dbImage, qImage, utmDb, utmQ = train_test_split(tImage, tutm, test_size=opt.query_ratio)
        numDb = len(dbImage)
        numQ = len(qImage)
        sio.savemat(os.path.join(self.root, 'train.mat'), {'whichSet': 'train', 'dataset': opt.dataset, 'dbImage': dbImage,
                                                      'qImage': qImage, 'utmDb': utmDb, 'numDb': numDb, 'utmQ': utmQ,
                                                      'numQ': numQ, 'posDistThr': opt.posDistThr,
                                                      'posDistSqThr': opt.posDistSqThr,
                                                      'nonTrivPosDistSqThr': opt.nonTrivPosDistSqThr})
        dbImage, qImage, utmDb, utmQ = train_test_split(vImage, vutm, test_size=opt.query_ratio)
        numDb = len(dbImage)
        numQ = len(qImage)
        sio.savemat(os.path.join(self.root, 'valid.mat'),
                    {'whichSet': 'val', 'dataset': opt.dataset, 'dbImage': dbImage,
                     'qImage': qImage, 'utmDb': utmDb, 'numDb': numDb, 'utmQ': utmQ,
                     'numQ': numQ, 'posDistThr': opt.posDistThr,
                     'posDistSqThr': opt.posDistSqThr, 'nonTrivPosDistSqThr': opt.nonTrivPosDistSqThr})
        self.extract_sp_feature()
        #self.get_useful_feature()

def main(opt):
    input_path=opt.src_root
    dics = sorted(os.listdir(input_path))
    with open(opt.Topo_path,'r') as f:
        data=json.load(f)['keyframes']
    root=opt.root
    db_path=os.path.join(root,'database')
    # q_path=os.path.join(root,'query')
    if not os.path.exists(db_path):
        os.makedirs(db_path)

    yaw_list=np.linspace(0,360,opt.yaw_num+1)[:-1]
    pitch_list=np.linspace(-opt.pitch_range/2,opt.pitch_range/2,opt.pitch_num)

    generator=Generate_data(dics,data,db_path,input_path,pitch_list,yaw_list,root)
    generator.start()
    generator.done=True
    generator.join()


if __name__ == '__main__':
    opt=options()
    main(opt)
