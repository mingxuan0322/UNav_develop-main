import numpy as np
import argparse
import json
import os
from sklearn.neighbors import NearestNeighbors
import logging
from scipy.sparse.csgraph import shortest_path
import h5py
from pathlib import Path
from multiprocessing import pool, cpu_count
from functools import partial

def options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--topomap_path', default=None, required=True,
                        help='path to topomap')
    parser.add_argument('--radius', default=100,type=int, required=True,
                        help='searching radius')
    parser.add_argument('--min_distance', default=100, type=int, required=True,
                        help='min distance to the boundaries')
    parser.add_argument('--cpu_num', default=cpu_count(), type=int, required=False,
                        help='cpu number')
    opt = parser.parse_args()
    return opt

def load_data(path):
    if os.path.exists(os.path.join(path, 'topo-map.json')):
        with open(os.path.join(path, 'topo-map.json'), 'r') as f:
            data = json.load(f)
    else:
        print('Topometric Map does not exists!')
        exit()
    return data

def ccw(A,B,C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

def judge_intersection(i, pts,positives,boundaries):
    boundary=boundaries[i]
    a,b=[boundary[0],boundary[1]],[boundary[2],boundary[3]]
    inter=[]
    for k,c in enumerate(pts):
        for j in range(len(positives[k])):
            d = pts[positives[k][j]]
            s = ''
            for kk in [c[0],c[1],d[0],d[1]]:
                s += str(kk).zfill(4)
            if ccw(a,c,d) != ccw(b,c,d) and ccw(a,b,c) != ccw(a,b,d):
                inter.append(s)
    s=''
    for k in boundary:
        s+=str(k).zfill(4)
    return {s:inter}

def find_path(pts,positives,distances,l):
    n=len(pts)
    M = np.zeros((n, n))
    for k,c in enumerate(pts):
        for j in range(len(positives[k])):
            d = pts[positives[k][j]]
            s = ''
            for kk in [c[0], c[1], d[0], d[1]]:
                s += str(kk).zfill(4)
            if l[s]==[]:
                M[k, positives[k][j]] = distances[k][j]
                M[positives[k][j], k] = M[k, positives[k][j]]
    _, Pr = shortest_path(M, directed=False, method='FW', return_predecessors=True)
    return Pr,M

def generate_world_map(pts,lines):
    knn = NearestNeighbors(n_jobs=1)
    knn.fit(pts)
    distances, positives = knn.radius_neighbors(pts, radius=opt.radius)
    l={}
    for k, c in enumerate(pts):
        for j in range(len(positives[k])):
            d = pts[positives[k][j]]
            s = ''
            for kk in [c[0], c[1], d[0], d[1]]:
                s += str(kk).zfill(4)
            l[s] = []
    if os.path.exists(intersection_dir)and(threshold==opt.min_distance):
        global b
        added_line=lines['ad']
        removed_line=lines['rm']
        for k,v in b.items():
            for vv in v:
                l[vv].append(k)
        for rm in removed_line:
            s=''
            for kk in rm:
                s += str(kk).zfill(4)
            line_list=b[s]
            b.pop(s)
            for line in line_list:
                l[line].remove(s)
        p = pool.Pool(processes=opt.cpu_num)
        n = len(added_line)
        f = partial(judge_intersection, pts=pts, positives=positives, boundaries=added_line)
        with p:
            results = list(p.imap(f, range(n)))
            for r in results:
                k,v=list(r.items())[0]
                b.update({k:v})
                for vv in v:
                    l[vv].append(k)
    else:
        print(f'============================\nstart judging intersection\n============================')
        b={}
        p = pool.Pool(processes=opt.cpu_num)
        n = len(lines)
        f = partial(judge_intersection,pts=pts,positives=positives, boundaries=lines)
        with p:
            results = list(p.imap(f, range(n)))
            for r in results:
                k,v=list(r.items())[0]
                b.update({k:v})
                for vv in v:
                    l[vv].append(k)
    logging.info("Finished Checking intersection")
    Pr,M=find_path(pts,positives,distances,l)
    return Pr,M,b

def get_path(Pr, i, j):
    path = [j]
    k = j
    while Pr[i, k] != -9999:
        path.append(Pr[i, k])
        k = Pr[i, k]
    return path[::-1]

def save(names,Pr,data,b,threshold):
    path = Path(path_dir)
    path.parent.mkdir(exist_ok=True, parents=True)
    file = h5py.File(str(path), 'w')
    grp = file.create_group('Path')
    for i,n in enumerate(names):
        grp.create_dataset(n, data=Pr[i])
        logging.info("Save Path: #%d"%i)
    file.close()
    with open(intersection_dir,'w') as f:
        json.dump({'data':data,'threshold':threshold,'boundaries':b},f)

def expand_boundaries(lines,thre):
    Lines=[]
    for l in lines:
        a,b=(l[0],l[1]),(l[2],l[3])
        if a[0] == b[0]:
            if a[1] > b[1]:
                t = a
                a = b
                b = t
            aa1 = (a[0] - thre, a[1] - thre)
            aa2 = (a[0] + thre, a[1] - thre)
            bb1 = (b[0] - thre, b[1] + thre)
            bb2 = (b[0] + thre, b[1] + thre)
        else:
            if a[0] > b[0]:
                t = a
                a = b
                b = t
            ang = -np.arctan((b[1] - a[1]) / (b[0] - a[0]))
            cos = np.cos(ang) * thre
            sin = np.sin(ang) * thre
            a = (int(a[0] - cos), int(a[1] + sin))
            b = (int(b[0] + cos), int(b[1] - sin))
            aa1 = (int(a[0] - sin), int(a[1] - cos))
            bb1 = (int(b[0] - sin), int(b[1] - cos))
            aa2 = (int(a[0] + sin), int(a[1] + cos))
            bb2 = (int(b[0] + sin), int(b[1] + cos))
        Lines.append([aa1[0],aa1[1],bb1[0],bb1[1]])
        Lines.append([aa2[0], aa2[1], bb2[0], bb2[1]])
        Lines.append([aa2[0], aa2[1], aa1[0], aa1[1]])
        Lines.append([bb1[0], bb1[1], bb2[0], bb2[1]])
    return Lines

def add_remove_determine(b,a):
    a = set(map(tuple, a))
    b = set(map(tuple, b))
    return list(map(list, b - a)),list(map(list, a-b))

def main():
    data = load_data(opt.topomap_path)
    keyframes=data['keyframes']
    pts=np.array([keyframes[k]['trans'] for k in list(keyframes.keys()) if k.split('_')[-1]=='00'],dtype=int)
    names=[k.split('_')[0] for k in list(keyframes.keys()) if k.split('_')[-1]=='00']
    with open(os.path.join(opt.topomap_path,'boundaries.json'),'r') as f:
        data=json.load(f)
        lines=data['lines']
        add_lines=data['add_lines']
        for i in add_lines:
            lines.append(i)
    global path_dir,intersection_dir
    path_dir = os.path.join(opt.topomap_path, 'path.h5')
    intersection_dir = os.path.join(opt.topomap_path, 'intersection.json')
    if os.path.exists(intersection_dir):
        with open(intersection_dir,'r') as f:
            d=json.load(f)
        global threshold,b
        b=d['boundaries']
        threshold=d['threshold']
        lines_ = d['data']['lines']
        add_lines = d['data']['add_lines']
        if (threshold==opt.min_distance):
            for i in add_lines:
                lines_.append(i)
            added_line,removed_line=add_remove_determine(lines,lines_)
            added_line=expand_boundaries(added_line,opt.min_distance)
            removed_line = expand_boundaries(removed_line, opt.min_distance)
            lines={'ad':added_line,'rm':removed_line}
        else:
            lines = expand_boundaries(lines, opt.min_distance)
    else:
        lines=expand_boundaries(lines,opt.min_distance)
    Pr,M,b=generate_world_map(pts,lines)
    save(names,Pr,data,b,opt.min_distance)

if __name__ == '__main__':
    opt = options()
    main()
