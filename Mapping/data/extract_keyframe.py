import argparse
import os
import msgpack
import numpy as np

def options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--maps', default=None, required=True,
                        help='path to maps')
    parser.add_argument('--src_dir', default=None, required=True,
                        help='path to src_dir')
    opt = parser.parse_args()
    return opt

def remove(opt):
    with open(opt.maps, "rb") as f:
        data = msgpack.unpackb(f.read(), use_list=False, raw=False)
    l=[]
    num1 = np.max([len(str(data['keyframes'][i]['src_frm_id'])) for i in data['keyframes']])
    list_dir = os.listdir(opt.src_dir)
    num2 = len(str(list_dir[0].replace('.png', '')))
    num = max(num1, num2)
    for i in data['keyframes']:
        t = str(data['keyframes'][i]['src_frm_id'] + 1).zfill(num)
        l.append(t)
    for i in list_dir:
        if i.replace('.png', '') not in l:
            os.remove(os.path.join(opt.src_dir, i))

opt=options()
remove(opt)