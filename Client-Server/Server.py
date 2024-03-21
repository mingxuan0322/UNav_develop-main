import socket
import threading
import numpy as np
import argparse
import json
import os
from os.path import join, exists, isfile
import natsort
import cv2
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import h5py
import sys
sys.path.append("../Localization")
import pytorch_NetVlad.netvlad as netvlad
from SuperPoint_SuperGlue.base_model import dynamic_load
from SuperPoint_SuperGlue import extractors
from SuperPoint_SuperGlue import matchers
from scipy.spatial.transform import Rotation as R
from scipy.spatial import KDTree
import pyimplicitdist
import poselib
import jpysocket
import time
from datetime import datetime
from skimage.measure import ransac
from skimage.transform import AffineTransform

def options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host_id', default=None, type=str, required=True,
                        help='host id')
    parser.add_argument('--port_id', default=None, type=int, required=True,
                        help='port id')
    parser.add_argument('--FloorPlan_scale', default=0.01, type=float, required=True,
                        help='Floor plan scale')
    parser.add_argument('--max_matches', default=50, type=int, required=False,
                        help='retrieval numbers')
    parser.add_argument('--topomap_path', default=None, required=True,
                        help='path of topomatric map')
    parser.add_argument('--database_path', default=None, required=True,
                        help='path of database')
    parser.add_argument('--max_matching_image_num', default=None, type=int,required=True,
                        help='maximum of matching data used by implicity model')
    parser.add_argument('--logs', default=None, required=True,
                        help='path to save trials')
    parser.add_argument('--Place', type=str, help='Place information')
    parser.add_argument('--Building', type=str, help='Building information')
    parser.add_argument('--Floor', type=str, help='Floor information')
    parser.add_argument('--nocuda', action='store_true', help='Dont use cuda')
    parser.add_argument('--cpu', action='store_true', help='cpu for global descriptors')
    parser.add_argument('--ckpt_path', type=str, default='vgg16_netvlad_checkpoint',
                        help='Path to load checkpoint from, for resuming training or testing.')
    parser.add_argument('--arch', type=str, default='vgg16',
                        help='basenetwork to use', choices=['vgg16', 'alexnet'])
    parser.add_argument('--vladv2', action='store_true', help='Use VLAD v2')
    parser.add_argument('--pooling', type=str, default='netvlad', help='type of pooling to use',
                        choices=['netvlad', 'max', 'avg'])
    parser.add_argument('--num_clusters', type=int, default=64, help='Number of NetVlad clusters. Default=64')
    opt = parser.parse_args()
    return opt

def tensor_from_names(opt, names, hfile):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    desc = [hfile[i]['global_descriptor'].__array__() for i in names]
    if opt.cpu:
        desc = torch.from_numpy(np.stack(desc, 0)).float()
    else:
        desc = torch.from_numpy(np.stack(desc, 0)).to(device).float()
    return desc

def load_data(opt):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    path = os.path.join(opt.topomap_path, opt.Place, opt.Building, opt.Floor)
    if os.path.exists(os.path.join(path, 'topo-map.json')):
        with open(os.path.join(path, 'topo-map.json'), 'r') as f:
            data = json.load(f)
        kf = data['keyframes']
        landmarks = data['landmarks']
        pts = np.array(
            [kf[k]['trans'] for k in list(kf.keys()) if k.split('_')[-1] == '00'], dtype=int)
        knames = [k.split('_')[0] for k in list(kf.keys()) if k.split('_')[-1] == '00']
        kf_name = natsort.natsorted(list(set(data['keyframes'].keys())))

        n = len(list(set([nn.split('_')[-1] for nn in kf_name])))
        kdtree = KDTree(pts)
        T = np.array(data['T'])
        rot_base = np.arctan2(T[1, 0], T[0, 0])

        path_file = h5py.File(os.path.join(path, 'path.h5'), 'r')['Path']

        data_path = os.path.join(opt.database_path, opt.Place, opt.Building, opt.Floor)
        hfile = h5py.File(os.path.join(data_path, "global_features.h5"), 'r')
        hfile_local = h5py.File(os.path.join(data_path, "feats-superpoint.h5"), 'r')

        names = []
        hfile.visititems(
            lambda _, obj: names.append(obj.parent.name.strip('/'))
            if isinstance(obj, h5py.Dataset) else None)

        kf_name = natsort.natsorted(list(set(data['keyframes'].keys())))
        db_names = [n for n in names if n.replace('.png', '') in kf_name]

        db_desc = tensor_from_names(opt, db_names, hfile)

        extractor = NetVladFeatureExtractor(opt.ckpt_path, arch=opt.arch, num_clusters=opt.num_clusters,
                                            pooling=opt.pooling, vladv2=opt.vladv2, nocuda=opt.nocuda)
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
            }}
        conf_match = {
            'output': 'matches-superglue',
            'model': {
                'name': 'superglue',
                'weights': 'outdoor',
                'sinkhorn_iterations': 50,
            },
        }

        Model_sp = dynamic_load(extractors, conf['model']['name'])
        sp_model = Model_sp(conf['model']).eval().to(device)
        Model_sg = dynamic_load(matchers, conf_match['model']['name'])
        sg_model = Model_sg(conf_match['model']).eval().to(device)
    else:
        print('Topometric Map does not exists!')
        exit()
    return [landmarks, knames, kdtree, rot_base, path_file, hfile_local, db_desc, extractor, sp_model, sg_model, n,
            db_names, kf,pts,T]

def input_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

class NetVladFeatureExtractor:
    def __init__(self, ckpt_path, arch='vgg16', num_clusters=64, pooling='netvlad', vladv2=False, nocuda=False,
                 input_transform=input_transform()):
        self.input_transform = input_transform

        flag_file = join(ckpt_path, 'checkpoints', 'flags.json')
        if exists(flag_file):
            with open(flag_file, 'r') as f:
                stored_flags = json.load(f)
                stored_num_clusters = stored_flags.get('num_clusters')
                if stored_num_clusters is not None:
                    num_clusters = stored_num_clusters
                    print(f'restore num_clusters to : {num_clusters}')
                stored_pooling = stored_flags.get('pooling')
                if stored_pooling is not None:
                    pooling = stored_pooling
                    print(f'restore pooling to : {pooling}')

        cuda = not nocuda
        if cuda and not torch.cuda.is_available():
            raise Exception("No GPU found, please run with --nocuda")

        self.device = torch.device("cuda" if cuda else "cpu")

        print('===> Building model')

        if arch.lower() == 'alexnet':
            encoder_dim = 256
            encoder = models.alexnet(pretrained=True)
            # capture only features and remove last relu and maxpool
            layers = list(encoder.features.children())[:-2]

            # if using pretrained only train conv5
            for l in layers[:-1]:
                for p in l.parameters():
                    p.requires_grad = False

        elif arch.lower() == 'vgg16':
            encoder_dim = 512
            encoder = models.vgg16(pretrained=True)
            # capture only feature part and remove last relu and maxpool
            layers = list(encoder.features.children())[:-2]

            # if using pretrained then only train conv5_1, conv5_2, and conv5_3
            for l in layers[:-5]:
                for p in l.parameters():
                    p.requires_grad = False

        encoder = nn.Sequential(*layers)
        self.model = nn.Module()
        self.model.add_module('encoder', encoder)

        if pooling.lower() == 'netvlad':
            net_vlad = netvlad.NetVLAD(num_clusters=num_clusters, dim=encoder_dim, vladv2=vladv2)
            self.model.add_module('pool', net_vlad)
        else:
            raise ValueError('Unknown pooling type: ' + pooling)

        resume_ckpt = join(ckpt_path, 'checkpoints', 'checkpoint.pth.tar')

        if isfile(resume_ckpt):
            print("=> loading checkpoint '{}'".format(resume_ckpt))
            checkpoint = torch.load(resume_ckpt, map_location=lambda storage, loc: storage)
            best_metric = checkpoint['best_score']
            self.model.load_state_dict(checkpoint['state_dict'], strict=False)
            self.model = self.model.eval().to(self.device)
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(resume_ckpt, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(resume_ckpt))

    def feature(self, image):
        if self.input_transform:
            image = self.input_transform(image)
            # batch size 1
            image = torch.stack([image])

        with torch.no_grad():
            input = image.to(self.device)
            image_encoding = self.model.encoder(input)
            vlad_encoding = self.model.pool(image_encoding)
            del input
            torch.cuda.empty_cache()
            return vlad_encoding.detach().cpu().numpy()

class Hloc():
    def __init__(self, opt, data, destinations):
        self.opt = opt
        self.landmarks, self.knames, self.kdtree, self.rot_base, self.path_file, self.hfile_local, self.db_desc, self.extractor, self.sp_model, self.sg_model, self.n, self.db_names, self.kf ,self.pts,self.T= \
            data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8], data[9], data[10],data[11], data[12],data[13], data[14]
        self.list_2d, self.list_3d, self.initial_poses, self.pps = [], [], [], []

        self.globs = ['.jpg', '.png', '.jpeg', '.JPG', '.PNG']

        self.destinations = destinations

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        r = 30
        self.gamma = 2 * np.pi / self.n
        self.theta = np.pi / 2 - np.pi / self.n
        self.rbar = r * np.sin(self.theta)
        self.rhat = r * np.cos(self.theta)

        self.x_, self.y_=-9999,-9999

    def prepare_data(self, image):

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # image = np.array(ImageOps.grayscale(image)).astype(np.float32)
        image = image[None]
        data = torch.from_numpy(image / 255.).unsqueeze(0)
        return data

    def extract_local_features(self, image0):
        data0 = self.prepare_data(image0)
        pred0 = self.sp_model(data0.to(self.device))
        del data0
        torch.cuda.empty_cache()
        pred0 = {k: v[0].cpu().detach().numpy() for k, v in pred0.items()}
        if 'keypoints' in pred0:
            pred0['keypoints'] = (pred0['keypoints'] + .5) - .5
        pred0.update({'image_size': np.array([image0.shape[0], image0.shape[1]])})
        return pred0

    def geometric_verification(self, i, feats0):
        feats1 = self.hfile_local[self.db_names[i]]
        data = {}
        for k in feats0.keys():
            data[k + '0'] = feats0[k]
        for k in feats0.keys():
            data[k + '1'] = feats1[k].__array__()
        data = {k: torch.from_numpy(v)[None].float().to(self.device)
                for k, v in data.items()}
        data['image0'] = torch.empty((1, 1,) + tuple(feats0['image_size'])[::-1])
        data['image1'] = torch.empty((1, 1,) + tuple(feats1['image_size'])[::-1])
        pred = self.sg_model(data)
        matches = pred['matches0'][0].detach().cpu().short().numpy()
        pts0, pts1, lms = [], [], []
        index_list = self.kf[self.db_names[i].replace('.png', '')]['kp_index']
        for n, m in enumerate(matches):
            if (m != -1) and (m in index_list):
                pts0.append(feats0['keypoints'][n].tolist())
                pts1.append(feats1['keypoints'][m].tolist())
                lms.append(self.kf[self.db_names[i].replace('.png', '')]['lm_ids'][index_list.index(m)])
        del data, feats1, pred
        tm = time.time()
        try:
            pts0_ = np.int32(pts0)
            pts1_ = np.int32(pts1)
            # F, mask = cv2.findFundamentalMat(pts0_, pts1_, cv2.RANSAC)
            _, inliers = ransac(
                (pts0_, pts1_),
                AffineTransform,
                min_samples=10,
                residual_threshold=20,
                max_trials=30)
            valid = sum(inliers)
            # valid = len(pts0_[mask.ravel() == 1])
        except:
            valid = 0
        torch.cuda.empty_cache()
        pt0, pt1 = pts0, pts1
        return [pt0, pt1, lms, valid, time.time() - tm]

    def colmap2world(self, tvec, quat):
        r = R.from_quat(quat)
        rmat = r.as_matrix()
        rmat = rmat.transpose()
        rot = R.from_matrix(r.as_matrix().transpose()).as_rotvec()
        return -np.matmul(rmat, tvec).reshape(3), rot

    def coarse_pose(self, kpts, lms, initial_pp):
        threshold = 6.0
        p2d = np.array(kpts)
        p2d_center = [x - initial_pp for x in p2d]
        p3d = np.array(
            [np.array([[self.landmarks[str(i)]['x']], [self.landmarks[str(i)]['y']], [self.landmarks[str(i)]['z']]],
                      dtype=float) for i in lms])
        poselib_pose, info = poselib.estimate_1D_radial_absolute_pose(p2d_center, p3d, {"max_reproj_error": threshold})
        p2d_inlier = p2d[info["inliers"]]
        p3d_inlier = p3d[info["inliers"]]
        initial_pose = pyimplicitdist.CameraPose()
        initial_pose.q_vec = poselib_pose.q
        initial_pose.t = poselib_pose.t
        out = pyimplicitdist.pose_refinement_1D_radial(p2d_inlier, p3d_inlier, initial_pose, initial_pp,
                                                       pyimplicitdist.PoseRefinement1DRadialOptions())
        return out, p2d_inlier, p3d_inlier

    def pose_refine(self, out, p2d_inlier, p3d_inlier):
        refined_initial_pose, pp = out['pose'], out['pp']
        cm_opt = pyimplicitdist.CostMatrixOptions()
        refinement_opt = pyimplicitdist.PoseRefinementOptions()
        cost_matrix = pyimplicitdist.build_cost_matrix(p2d_inlier, cm_opt, pp)
        pose = pyimplicitdist.pose_refinement(p2d_inlier, p3d_inlier, cost_matrix, pp, refined_initial_pose,
                                              refinement_opt)
        qvec = pose.q_vec
        tvec = pose.t
        qvec = [qvec[1], qvec[2], qvec[3], qvec[0]]
        tvec, qvec = self.colmap2world(tvec, qvec)
        return tvec, qvec

    def pose_multi_refine(self, list_2d, list_3d, initial_poses, pps):
        cm_opt = pyimplicitdist.CostMatrixOptions()
        refinement_opt = pyimplicitdist.PoseRefinementOptions()
        invalid_id, list_2d_valid, list_3d_valid, initial_poses_valid, pps_valid = [], [], [], [], []
        for i in range(len(list_2d)):
            if isinstance(pps[i], str):
                invalid_id.append(i)
            else:
                list_2d_valid.append(list_2d[i])
                list_3d_valid.append(list_3d[i])
                initial_poses_valid.append(initial_poses[i])
                pps_valid.append(pps[i])
        cost_matrix = pyimplicitdist.build_cost_matrix_multi(list_2d_valid, cm_opt, np.average(pps_valid, 0))
        poses_valid = pyimplicitdist.pose_refinement_multi(list_2d_valid, list_3d_valid, cost_matrix,
                                                           np.average(pps_valid, 0), initial_poses_valid,
                                                           refinement_opt)
        qvecs = []
        tvecs = []
        j = 0
        for i in range(len(list_2d)):
            if i not in invalid_id:
                qvec = poses_valid[j].q_vec
                tvec = poses_valid[j].t
                qvec = [qvec[1], qvec[2], qvec[3], qvec[0]]
                tvec, qvec = self.colmap2world(tvec, qvec)
                qvecs.append(qvec)
                tvecs.append(tvec)
                j += 1
            else:
                qvecs.append('None')
                tvecs.append('None')
        return tvecs, qvecs

    def get_start(self, Pr, x, y):
        _, i_ = self.kdtree.query((x, y), k=10)
        min_ang = np.inf
        i = None
        distance=0
        for index in i_:
            x0, y0 = self.pts[index]
            if Pr[index] != -9999:
                x1, y1 = self.pts[Pr[index]]
                rot0 = np.arctan2(x - x0, y - y0)
                rot1 = np.arctan2(x0 - x1, y0 - y1)
                distance = abs(rot0 - rot1)
                if min_ang > distance:
                    min_ang = distance
                    i = index
        return i,distance

    def get_path(self, Pr, j):
        paths = [self.knames[j]]
        k = j
        while Pr[k] != -9999:
            paths.append(self.knames[Pr[k]])
            k = Pr[k]
        return paths

    def cloest_path(self, destination_id, x, y):
        indexs,distance0 = self.get_start(self.path_file[destination_id], x[0], y[0])
        try:
            paths = self.get_path(self.path_file[destination_id], indexs)
        except:
            paths = []
        return paths,distance0

    def action(self, rot_ang, distance):
        print(distance * self.opt.FloorPlan_scale * 0.3048)
        rot_ang=(-rot_ang)%360
        rot_clock=round(rot_ang/30)%12
        if rot_clock==0:
            rot_clock=12
        message='Please walk %.1f meters along %d clock\n'% (
                distance * self.opt.FloorPlan_scale * 0.3048,rot_clock)
        return message

    def run(self, image, destination_id):
        # tm = time.time()
        self.query_desc = self.extractor.feature(image)[0]
        if self.opt.cpu:
            self.query_desc = torch.from_numpy(self.query_desc).unsqueeze(0).float()
        else:
            self.query_desc = torch.from_numpy(self.query_desc).unsqueeze(0).to(self.device).float()
        sim = torch.einsum('id,jd->ij', self.query_desc, self.db_desc)
        topk = torch.topk(sim, self.opt.max_matches, dim=1).indices.cpu().numpy()
        # print('retrieval:', time.time() - tm)
        # tm=time.time()
        feats0 = self.extract_local_features(image)
        kp, lm = [], []
        tt = 0
        for i in topk[0]:
            pt0, pt1, lms, valid, ttt = self.geometric_verification(i, feats0)
            tt += ttt
            if valid > 30:
                for j in range(len(lms)):
                    kp.append(pt0[j])
                    lm.append(lms[j])
        del self.query_desc, feats0
        # print('geo:',tt)
        # print('geo_total:', time.time() - tm)
        torch.cuda.empty_cache()
        # current_location=-1
        if len(kp) > 0:
            height, width, _ = image.shape
            out, p2d_inlier, p3d_inlier = self.coarse_pose(kp, lm, np.array([width / 2, height / 2]))
            self.list_2d.append(p2d_inlier)
            self.list_3d.append(p3d_inlier)
            self.initial_poses.append(out['pose'])
            self.pps.append(out['pp'])
            if len(self.list_2d)>self.opt.max_matching_image_num:
                self.list_2d.pop(0)
                self.list_3d.pop(0)
                self.initial_poses.pop(0)
                self.pps.pop(0)
            tvecs, qvecs = self.pose_multi_refine(self.list_2d, self.list_3d, self.initial_poses, self.pps)
            tvec, qvec = tvecs[-1], qvecs[-1]
            x_, _, y_ = tvec
            ang = -qvec[1] - self.rot_base
            tvec = self.T @ np.array([[x_], [y_], [1]])
            x_, y_ = tvec
            print("Estimated location: x: %d, y: %d, ang: %d" % (x_, y_, ang * 180 / np.pi))
            # current_location = "_%d_%d_%d.png" % (x_, y_, ang * 180 / np.pi)
            self.x_, self.y_=x_,y_
            paths,distance_ = self.cloest_path(destination_id, x_, y_)
            if len(paths) > 1:
                x0, y0 = self.pts[self.knames.index(paths[0])]
                l0 = np.linalg.norm([x_ - x0, y_ - y0])
                if (l0 < 20):
                    paths.pop(0)
                x0, y0 = self.pts[self.knames.index(paths[0])]
                distance = np.linalg.norm([x_ - x0, y_ - y0])
                distance_+=distance
                rot = np.arctan2(x_ - x0, y_ - y0)
                rot_ang = (rot - ang) / np.pi * 180
                message = self.action(rot_ang[0], distance)
            else:
                message = 'The path to this destination has been blocked, please contact the map manager to update it\n'
            distance_=distance_* self.opt.FloorPlan_scale * 0.3048
            if (distance_<1)and(len(paths)<=1):
                message='You have arrived your destination'
        else:
            print('Cannot localize at this point, please take some steps or turn around')
            message = 'Cannot localize at this point, please take some steps or turn around\n'
        return message, [self.x_.tolist()[0], self.y_.tolist()[0]]

class Client(threading.Thread):
    def __init__(self, socket, address, id, name, signal, hloc, connections, destinations, destination_dicts,
                 logs):
        threading.Thread.__init__(self)
        self.socket = socket
        self.address = address
        self.id = id
        self.name = name
        self.signal = signal
        self.connections = connections
        self.total_connections = 0
        self.hloc = hloc
        self.destination = destinations
        self.destination_dicts = destination_dicts
        self.logs=logs
        if os.path.exists(logs):
            with open(logs,'r') as f:
                self.record=json.load(f)
                self.trial=str(len(self.record)).zfill(5)
        else:
            self.record={}
            self.trial='00000'

    def __str__(self):
        return str(self.id) + " " + str(self.address)

    def recvall(self, sock, count):
        buf = b''
        while count:
            newbuf = sock.recv(count)
            if not newbuf: return None
            buf += newbuf
            count -= len(newbuf)
        return buf

    def date(self,s):
        return [s.year,s.month,s.day,s.hour,s.minute,s.second]

    def run(self):
        while self.signal:
            # try:
                number = self.recvall(self.socket, 4)
                if not number:
                    continue
                command = int.from_bytes(number, 'big')
                if command == 1:
                    print('=========================')
                    tm = time.time()
                    length = self.recvall(self.socket, 4)
                    data = self.recvall(self.socket, int.from_bytes(length, 'big'))
                    if not data:
                        continue
                    nparr = np.frombuffer(data, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    print("Received image")

                    Destination = self.socket.recv(4096)
                    Destination = jpysocket.jpydecode(Destination)
                    Place, Building, Floor, Destination_ = Destination.split(',')
                    dicts = self.destination[Place][Building][Floor]
                    for i in dicts:
                        for k, v in i.items():
                            if k == Destination_:
                                destination = v
                                break
                    print('receive image cost:%f seconds' % (time.time() - tm))
                    # test the existing user images
                    # path = os.path.join(self.usr_image, Place, Building, Floor)
                    # img_list = os.listdir(path)
                    # import random
                    # im = os.path.join(path, random.choice(img_list))
                    # print('test image: ', im)
                    # img = cv2.imread(im)
                    tm = time.time()
                    message, current_loc = self.hloc.run(img, destination)
                    self.record[self.trial].update({'stop time':self.date(datetime.now())})
                    self.waypoints.append(current_loc)
                    self.record[self.trial].update({'waypoints':self.waypoints})
                    self.record[self.trial].update({'destination': [Place,Building,Floor,self.hloc.pts[self.hloc.knames.index(destination)].tolist()]})
                    with open(self.logs,'w') as f:
                        json.dump(self.record,f)
                    print('localization cost:%f seconds' % (time.time() - tm))
                    self.socket.send(bytes(message, 'UTF-8'))
                    print(message)
                    print('use %d images'%len(self.hloc.list_2d))
                    # if current_loc!=None:
                    #     if not os.path.exists(path):
                    #         os.makedirs(path)
                    #         n = 0
                    #     else:
                    #         n = len(os.listdir(path))
                    #     st=str(n).zfill(5)+current_loc
                    #     print(st)
                    #     cv2.imwrite(os.path.join(path,st),img)

                elif command == 0:
                    destination_dicts = str(self.destination_dicts) + '\n'
                    self.socket.send(bytes(destination_dicts, 'UTF-8'))
                    self.record[self.trial] = {'start time': self.date(datetime.now())}
                    self.waypoints = []
            # except:
            #     print("Client " + str(self.address) + " has disconnected")
            #     self.signal = False
            #     self.connections.remove(self)
            #     break

def newConnections(socket, connections, total_connections, data, destinations, destination_dicts, opt):
    while True:
        sock, address = socket.accept()
        hloc = Hloc(opt, data, destinations)
        connections.append(
            Client(sock, address, total_connections, "Name", True, hloc, connections, destinations, destination_dicts,
                   opt.logs))
        connections[len(connections) - 1].start()
        print("New connection at ID " + str(connections[len(connections) - 1]))
        total_connections += 1

def destination_information(path):
    if os.path.exists(path):
        with open(path, 'r') as f:
            destinations = json.load(f)
        destinations_dicts = {}
        for k0, v0 in destinations.items():
            building_dicts = {}
            for k1, v1 in v0.items():
                floor_dicts = {}
                for k2, v2 in v1.items():
                    list0 = []
                    for v3 in v2:
                        list0.append(list(v3.keys())[0])
                    floor_dicts.update({k2: list0})
                building_dicts.update({k1: floor_dicts})
            destinations_dicts.update({k0: building_dicts})
    else:
        print("destination file does not exist")
        exit()
    return destinations, destinations_dicts

def main():
    opt = options()

    host = opt.host_id
    port = opt.port_id

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind((host, port))
    sock.listen(5)

    connections = []
    total_connections = 0

    destinations, destination_dicts = destination_information(os.path.join(opt.topomap_path, 'destination.json'))
    data = load_data(opt)
    newConnectionsThread = threading.Thread(target=newConnections, args=(
        sock, connections, total_connections, data, destinations, destination_dicts, opt))
    newConnectionsThread.start()

if __name__ == '__main__':
    main()
