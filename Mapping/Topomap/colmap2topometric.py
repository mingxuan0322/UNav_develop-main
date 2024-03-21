import argparse
import json
import os
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import collections
import natsort
import struct
import numpy as np
import logging

def options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--maps', default=None, required=True,
                        help='path to map')
    parser.add_argument('--outf', default=None, required=True,
                        help='path to outf')
    opt = parser.parse_args()
    return opt

def slam2world(t, r):
    r = R.from_quat(r)
    return -np.matmul(r.as_matrix().transpose(), t)

class get_colmap_data():
    def __init__(self,opt):
        self.images_path=os.path.join(opt.maps,'images.bin')
        self.point3d_path=os.path.join(opt.maps,'points3D.bin')

    def read_next_bytes(self,fid, num_bytes, format_char_sequence, endian_character="<"):
        data = fid.read(num_bytes)
        return struct.unpack(endian_character + format_char_sequence, data)

    def read_points3d_binary(self,path_to_model_file):
        points3D = {}
        with open(path_to_model_file, "rb") as fid:
            num_points = self.read_next_bytes(fid, 8, "Q")[0]
            for point_line_index in range(num_points):
                binary_point_line_properties = self.read_next_bytes(
                    fid, num_bytes=43, format_char_sequence="QdddBBBd")
                point3D_id = binary_point_line_properties[0]
                xyz = np.array(binary_point_line_properties[1:4])
                rgb = np.array(binary_point_line_properties[4:7])
                error = np.array(binary_point_line_properties[7])
                track_length = self.read_next_bytes(
                    fid, num_bytes=8, format_char_sequence="Q")[0]
                track_elems = self.read_next_bytes(
                    fid, num_bytes=8 * track_length,
                    format_char_sequence="ii" * track_length)
                image_ids = np.array(tuple(map(int, track_elems[0::2])))
                point2D_idxs = np.array(tuple(map(int, track_elems[1::2])))
                points3D[point3D_id] = {'id': point3D_id, 'xyz': xyz, 'rgb': rgb,
                                        'error': error, 'image_ids': image_ids,
                                        'point2D_idxs': point2D_idxs}
        return points3D

    def read_images_binary(self,path_to_model_file):
        images = {}
        with open(path_to_model_file, "rb") as fid:
            num_reg_images = self.read_next_bytes(fid, 8, "Q")[0]
            for image_index in range(num_reg_images):
                binary_image_properties = self.read_next_bytes(
                    fid, num_bytes=64, format_char_sequence="idddddddi")
                image_id = binary_image_properties[0]
                qvec = np.array(binary_image_properties[1:5])
                tvec = np.array(binary_image_properties[5:8])
                camera_id = binary_image_properties[8]
                image_name = ""
                current_char = self.read_next_bytes(fid, 1, "c")[0]
                while current_char != b"\x00":
                    image_name += current_char.decode("utf-8")
                    current_char = self.read_next_bytes(fid, 1, "c")[0]
                num_points2D = self.read_next_bytes(fid, num_bytes=8,
                                               format_char_sequence="Q")[0]
                x_y_id_s = self.read_next_bytes(fid, num_bytes=24 * num_points2D,
                                           format_char_sequence="ddq" * num_points2D)
                xys = np.column_stack([tuple(map(float, x_y_id_s[0::3])),
                                       tuple(map(float, x_y_id_s[1::3]))])
                point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
                images[image_id] = {'id': image_id, 'qvec': qvec, 'tvec': tvec,
                                    'camera_id': camera_id, 'name': image_name,
                                    'xys': xys, 'point3D_ids': point3D_ids}
        return images

    def load(self):
        images = self.read_images_binary(self.images_path)
        images=collections.OrderedDict(natsort.natsorted(images.items()))
        P3D = self.read_points3d_binary(self.point3d_path)
        ffid={}
        key = []
        point3d = []
        for id, point in P3D.items():
            pos = point["xyz"]
            point3d.append([pos[0], 0, pos[2]])
        for id, point in images.items():
            trans = point["tvec"]
            rot = point['qvec']
            rot=[rot[1],rot[2],rot[3],rot[0]]
            pos = slam2world(trans, rot)
            key.append([pos[0], pos[2]])
        kf = np.array(key)
        source = o3d.geometry.PointCloud()
        source.points = o3d.utility.Vector3dVector(point3d)
        cl, source1 = source.remove_radius_outlier(nb_points=16, radius=0.25)
        source1 = source.select_by_index(source1)
        features = np.array(source1.points)
        features = np.vstack((features[:, 0], features[:, 2])).T
        for i in list(images.keys()):
            t = images[i]['name']
            im = opt.src_dir + '/' + t
            jj = images[i]['point3D_ids']
            j = [int(ii) for ii in jj if ii != -1]
            lm = [P3D[i]['xyz'].tolist() for i in j]
            kp = images[i]['xys']
            rot = images[i]['qvec']
            rot = [rot[1], rot[2], rot[3], rot[0]]
            gp = []
            index = [ii for ii, x in enumerate(images[i]['point3D_ids']) if x != -1]
            for ii, tt in enumerate(kp):
                if ii in index:
                    gp.append(np.array(tt))
            ffid.update({t.replace('.png',''): {'frame': im, 'ang': R.from_quat(rot).as_rotvec()[1], 'lm': lm, 'lm_id': j,
                             'gp': np.array(gp).tolist()}})
        return ffid, features.tolist(), kf.tolist()

def transformer(opt,data):
    T=np.array(data['T'])
    kf = {}
    lm = {}
    loader = get_colmap_data(opt)
    images = loader.read_images_binary(loader.images_path)
    images = collections.OrderedDict(natsort.natsorted(images.items()))
    P3D = loader.read_points3d_binary(loader.point3d_path)
    point3d = []
    Z = []
    for id, point in P3D.items():
        pos = point["xyz"]
        point3d.append([pos[0], 0, pos[2]])
        Z.append(pos)
    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(point3d)
    cl, source1 = source.remove_radius_outlier(nb_points=16, radius=0.25)
    invalid = []
    for i, index in enumerate(P3D.keys()):
        logging.info("generate landmarks:\t{}".format(index))
        if i in source1:
            lm.update({str(index): {'x': Z[i][0], 'y': Z[i][1], 'z': Z[i][2]}})
        else:
            invalid.append(str(index))
    for j, i in enumerate(list(images.keys())):
        logging.info("generate keyframes:\t{}".format(i))
        k = images[i]['tvec']
        rot = images[i]['qvec']
        rot = [rot[1], rot[2], rot[3], rot[0]]
        pos = slam2world(k, rot)
        t_fp = np.array([pos[0], pos[2], 1]).T
        t_mp = ((T @ t_fp).T).tolist()
        kps, kes, klm, ids = [], [], [], []
        for ind, k in enumerate(images[i]['point3D_ids']):
            if k != -1:
                if str(k) not in invalid:
                    ids.append(ind)
                    kps.append(images[i]['xys'][ind].tolist())
                    klm.append(int(k))
        kf.update({images[i]['name'].replace('.png',''): {'keypts': kps, 'lm_ids': klm, 'kp_index': ids, 'trans': t_mp,
                                  'rot': R.from_quat(rot).as_rotvec()[1] - data['base_rot']}})
    return {'landmarks': lm, 'keyframes': kf, 'floorplan': data['floorplan'], 'T': T.tolist()}

def main(opt):
    topo_path=os.path.join(opt.outf,'slam_data.json')
    with open(topo_path, 'r') as f:
        data = json.load(f)
    data=transformer(opt,data)
    with open(os.path.join(opt.outf, 'topo-map.json'), 'w') as f:
        json.dump(data, f)

if __name__ == '__main__':
    opt = options()
    main(opt)