from tqdm import tqdm
import numpy as np
import os, json
from PIL import Image, ImageDraw

def GT_data(GT_path, topo_path, plan_path, demo_out):
    fp = open(GT_path)
    with open(topo_path, 'r') as f:
        data = json.load(f)
    plan = Image.open(plan_path)
    width, height = plan.size
    plot_scale = width / 3400
    draw = ImageDraw.Draw(plan)
    T = np.array(data['T'])
    lines = []
    for i, line in enumerate(fp):
        if (i > 50):
            line_ = line.replace('\n', '').split(' ')
            if line[0] not in ['[', 'm']:
                lines.append(line_)
    fp.close()
    data_ = {}
    for i in tqdm(range(int(len(lines) / 7))):
        id = lines[i * 7][0]
        r = lines[(i * 7 + 1):(i * 7 + 4)]
        rot_cw = []
        for row in r:
            cc = []
            for column in row:
                if column != '':
                    cc.append(float(column))
            rot_cw.append(np.array(cc))
        trans_cw = []
        t = lines[(i * 7 + 4):(i * 7 + 7)]
        for row in t:
            cc = []
            for column in row:
                if column != '':
                    cc.append(float(column))
            trans_cw.append(np.array(cc))
        rot_cw = np.array(rot_cw)
        trans_cw = np.array(trans_cw)
        trans_wc = -np.matmul(rot_cw.T, trans_cw).reshape(3)
        data_[id] = np.array(trans_wc)
    for k, trans_wc in data_.items():
        x_, _, y_ = trans_wc
        tvec = T @ np.array([[x_], [y_], [1]])
        x_, y_ = tvec
        draw.ellipse(
            (x_ - 5 * plot_scale, y_ - 5 * plot_scale, x_ + 5 * plot_scale, y_ + 5 * plot_scale),
            fill=(50, 0, 106))
    print(len(data_))
    plan.save(demo_out)


GT_path, topo_path, plan_path, demo_out = '/media/endeleze/Endeleze_5T1/UNav/Mapping/data/maps/New_York_City/LightHouse/mapAB.txt', os.path.join(
    '/home/endeleze/Desktop/UNav_develop/Mapping',
    'topo-map.json'), '/media/endeleze/Endeleze_5T/UNav/Mapping/data/floor_plan/New_York_City/LightHouse/3.jpg', '/home/endeleze/Desktop/UNav_develop/Mapping/test.png'

GT_data(GT_path, topo_path, plan_path, demo_out)
