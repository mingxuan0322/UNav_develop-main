from __future__ import print_function
import argparse
from types import SimpleNamespace
import cv2
import random, shutil, json
from os.path import join, exists, isfile, realpath, dirname

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.nn import Flatten
import h5py
from pathlib import Path
from PIL import Image,ImageOps
from SuperPoint_SuperGlue.tools import map_tensor
from tqdm import tqdm
import numpy as np
import pytorch_NetVlad.netvlad as netvlad
from SuperPoint_SuperGlue.base_model import dynamic_load
from SuperPoint_SuperGlue import extractors

parser = argparse.ArgumentParser(description='pytorch-NetVlad')
parser.add_argument('--nocuda', action='store_true', help='Dont use cuda')
parser.add_argument('--superpoint_local', action='store_true', help='extract local feature')
parser.add_argument('--ckpt_path', type=str, default='vgg16_netvlad_checkpoint', help='Path to load checkpoint from, for resuming training or testing.')
parser.add_argument('--arch', type=str, default='vgg16', 
        help='basenetwork to use', choices=['vgg16', 'alexnet'])
parser.add_argument('--vladv2', action='store_true', help='Use VLAD v2')
parser.add_argument('--pooling', type=str, default='netvlad', help='type of pooling to use',
        choices=['netvlad', 'max', 'avg'])
parser.add_argument('--num_clusters', type=int, default=64, help='Number of NetVlad clusters. Default=64')
parser.add_argument('--image_dir', type=Path)
parser.add_argument('--output', type=Path, default='output',help='Output folder for featues')

def input_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
    ])

class NetVladFeatureExtractor:
    def __init__ (self, ckpt_path, arch='vgg16', num_clusters=64, pooling='netvlad', vladv2=False, nocuda=False, input_transform=input_transform()):
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
            #batch size 1
            image = torch.stack([image])
        
        with torch.no_grad():
            input = image.to(self.device)
            image_encoding = self.model.encoder(input)
            vlad_encoding = self.model.pool(image_encoding) 

            return vlad_encoding.detach().cpu().numpy()

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

def prepare_data(image):
    image = np.array(ImageOps.grayscale(image)).astype(np.float32)
    image = image[None]
    data = torch.from_numpy(image / 255.).unsqueeze(0)
    return data

def extract_local_feature(args):
    conf = {
        'output': 'feats-superpoint',
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
    cuda = not args.nocuda
    device = torch.device("cuda" if cuda else "cpu")
    Model = dynamic_load(extractors, conf['model']['name'])
    model = Model(conf['model']).eval().to(device)

    loader = ImageDataset(args.image_dir, conf['preprocessing'])
    loader = torch.utils.data.DataLoader(loader, num_workers=1)

    feature_path = Path(args.output, conf['output'] + '.h5')
    feature_path.parent.mkdir(exist_ok=True, parents=True)
    feature_file = h5py.File(str(feature_path), 'a')

    for data in tqdm(loader,desc='extract local features'):
        pred = model(map_tensor(data, lambda x: x.to(device)))
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


if __name__ == "__main__":
    args = parser.parse_args()

    globs=['*.jpg', '*.png', '*.jpeg', '*.JPG', '*.PNG']
    image_paths = []
    for g in globs:
        image_paths += list(Path(args.image_dir).glob('**/'+g))
    if len(image_paths) == 0:
        raise ValueError(f'Could not find any image in root: {root}.')
    image_paths = sorted(list(set(image_paths)))
    image_paths = [i.relative_to(args.image_dir) for i in image_paths]
    
    global_feature_path=Path(args.output, 'global_features.h5')
    global_feature_path.parent.mkdir(exist_ok=True, parents=True)
    global_feature_file = h5py.File(str(global_feature_path), 'w')

    extractor = NetVladFeatureExtractor(args.ckpt_path, arch=args.arch, num_clusters=args.num_clusters, pooling=args.pooling, vladv2=args.vladv2, nocuda=args.nocuda)

    cnt = 0
    for im_file in tqdm(image_paths,desc='extract global features'):
        image = Image.open(str(args.image_dir / im_file))
        feature = extractor.feature(image)[0]
        #print(f'shape {feature.shape}')
        grp = global_feature_file.create_group(str(im_file))
        grp.create_dataset('global_descriptor', data=feature)

    global_feature_file.close()

    if args.superpoint_local:
        extract_local_feature(args)

