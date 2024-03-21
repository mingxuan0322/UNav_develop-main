import numpy as np
import argparse
import pygsheets
import re
import matplotlib as mpl
from matplotlib import image
import plotly.graph_objects as go
import os
import json
from PIL import Image
import glob
import shutil
from tqdm import tqdm


def options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--floorplan', type=str, help='Floor Plan path')
    parser.add_argument('--topomap_path', type=str, help='Topometric map path')
    parser.add_argument('--Output_path', type=str, help='Output path')
    parser.add_argument('--FloorPlan_scale', default=0.01, type=float, required=True,
                        help='Floor plan scale')
    parser.add_argument('--service_file', default=None, required=True,
                        help='path to googlesheet file')
    parser.add_argument('--sheet_name', default=None, required=True,
                        help='sheet name')
    parser.add_argument('--sheet', default=None, type=int, required=True,
                        help='sheet number')
    parser.add_argument('--read_start', default=None, type=str, required=True,
                        help='read ground truth from')
    parser.add_argument('--read_end', default=None, type=str, required=True,
                        help='read ground truth to')
    parser.add_argument('--sample_rate', default=None, type=int, nargs="+", required=True,
                        help='sample rate')
    parser.add_argument('--rot_rate', default=None, type=int, required=True,
                        help='rot rate')
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


class heat_map():
    def __init__(self, opt):
        self.opt = opt
        self.floorplan = opt.floorplan
        self.gc = pygsheets.authorize(service_file=opt.service_file)
        self.sh = self.gc.open(self.opt.sheet_name)
        self.wks = self.sh[self.opt.sheet]
        self.key_frames = load_data(self.opt.topomap_path)['keyframes']

    def colorFader(self, c1, c2, mix=0):
        c1 = np.array(mpl.colors.to_rgb(c1))
        c2 = np.array(mpl.colors.to_rgb(c2))
        return mpl.colors.to_hex((1 - mix) * c1 + mix * c2)

    def load(self, read_start, read_end):
        return np.array(self.wks.get_values_batch([read_start + ':' + read_end])).squeeze().astype(np.int)

    def load_data(self, start, end):
        GT = self.load(start, end)
        column, row, _ = re.split('(\d+)', start)
        estimation = []
        for sprt in self.opt.sample_rate:
            column_new = self.new_column(column, (sprt - 1) * 48 + (self.opt.rot_rate - 1) * 8 + 3)
            new_start = column_new + row
            _, row_, _ = re.split('(\d+)', end)
            column_new = self.new_column(column_new, 1)
            new_end = column_new + row_
            XY = self.load(new_start, new_end)
            column_new = self.new_column(column_new, 3)
            new_start = column_new + row
            new_end = column_new + row_
            estimation.append(np.hstack((XY, self.load(new_start, new_end).reshape(-1, 1))))
        return GT, estimation

    def new_column(self, column, num):
        asc_src = 0
        s = 1
        while len(column) > 0:
            asc_src += (ord(column[-1]) - 64) * s
            column = column[:-1]
            s *= 26
        asc_column = asc_src + num
        l = ''
        while asc_column > 0:
            res = asc_column % 26
            if res == 0:
                res = 26
                asc_column -= 1
            asc_column = asc_column // 26
            l = chr(res + 64) + l
        return l

    def error(self, GT, estimation):
        error_t = np.linalg.norm(GT[:, :-1] - estimation[:, :-1], axis=1) * opt.FloorPlan_scale * 3.28084
        ang = []
        for i in abs(np.array(GT[:, -1]) - estimation[:, -1]):
            ang.append(min(i, 360 - i))
        error_r = np.array(ang)
        return np.vstack((error_t, error_r))

    def plot_data(self, h, GT, error, estimation, sample_rate, t):
        x, y = [], []
        for i, index in enumerate(self.key_frames.keys()):
            if i % sample_rate == 0:
                k = self.key_frames[index]
                x_, y_ = k['trans']
                x.append(x_)
                y.append(h - y_)
        estx = [i for i in estimation[:, 0]]
        esty = [h - i for i in estimation[:, 1]]
        error_vpr = list(error[0, :])
        if t:
            estx.append(0)
            esty.append(0)
            error_vpr.append(0)
        size = [int((i - min(error_vpr)) * 95 / (max(error_vpr) - min(error_vpr))) + 15 for i in error_vpr]
        data = [
            go.Scatter(name='reference image', x=x, y=y, mode='markers',
                       marker=dict(color='lightskyblue', symbol=316, line=dict(width=1), size=10)),
            go.Scatter(name='query image (GT)', x=GT[:, 0], y=[h - i for i in GT[:, 1]], mode='markers+text',
                       text=[str(i) for i in range(len(GT))],
                       textfont=dict(family="Times New Roman", size=50, color="black"),
                       marker=dict(color='black', symbol='circle-open', line=dict(width=10), size=60)),
            go.Scatter(name='query image (estimate)', x=estx, y=esty,
                       mode='markers',
                       marker=dict(color=error_vpr, size=size, showscale=True,
                                   colorscale=[[0, 'rgb(0,200,0)'],
                                               [0.5, 'rgb(255,165,0)'],
                                               [1, 'rgb(255,0,0)']])),
            go.Scatter(x=estx, y=esty,
                       mode='markers',
                       marker=dict(color='black', size=size, symbol='circle-open', showscale=False,line=dict(width=5)))
        ]
        if t:
            data.append(go.Scatter(x=[500], y=[h-300], mode='text',
                                   text=['x/' + str(sample_rate)],
                                   textfont=dict(family="Times New Roman", size=150, color="black")
                                   ))
        for i in range(len(GT)):
            data.append(go.Scatter(x=[GT[i, 0], estimation[i, 0]], y=[h - GT[i, 1], h - estimation[i, 1]], mode='lines',
                                   line={'dash': 'dash', 'color': 'black', 'width': 3}))
        return data

    def create_image(self, h, w, GT, error, estimation, sample_rate, map_path, k):
        fig = go.Figure(data=self.plot_data(h, GT, error, estimation, sample_rate, k))
        fig.add_layout_image(
            dict(source=map_path, xref="x",
                 yref="y", x=0, y=h, sizex=w, sizey=h, sizing="stretch", opacity=1, layer="below"))
        fig.update_layout(template="plotly_white", autosize=False, width=w, height=h, xaxis_range=[300, w - 300],
                          yaxis_range=[0, h],
                          xaxis={'showgrid': False, 'visible': False, }, yaxis={'showgrid': False, 'visible': False, },
                          legend=dict(yanchor="top", y=0.15, xanchor="left", x=0.05,
                                      title_font_family="Times New Roman", font=dict(size=45, family="Times New Roman"),
                                      bgcolor="white",
                                      bordercolor="Black",
                                      borderwidth=2),
                          margin=dict(l=0, r=0, b=0, t=0, pad=0), paper_bgcolor="White",
                          font=dict(family="Times New Roman", size=40),
                          )
        for trace in fig['data']:
            if (trace['name'] != 'reference image') and (trace['name'] != 'query image (GT)') and (
                    trace['name'] != 'query image (estimate)'): trace['showlegend'] = False
        return fig

    def draw_map(self, GT, error, estimation):
        map_path = self.opt.floorplan
        img = image.imread(map_path)
        h, w, _ = img.shape
        if len(estimation) == 1:
            fig = self.create_image(h, w, GT, error[0], estimation[0], self.opt.sample_rate[0], map_path, False)
            outpath = os.path.join(os.path.split(self.opt.Output_path)[0])
            if not os.path.exists(outpath):
                os.makedirs(outpath)
            fig.write_image(self.opt.Output_path)
        else:
            outpath = os.path.join(os.path.split(self.opt.Output_path)[0])
            outemp = os.path.join(outpath, 'images')
            if not os.path.exists(outemp):
                os.makedirs(outemp)
            for i, k in enumerate(tqdm(self.opt.sample_rate)):
                fig = self.create_image(h, w, GT, error[i], estimation[i], k, map_path, True)
                fig.write_image(os.path.join(outemp, str(i) + '.png'))
            frames = [Image.open(image) for image in glob.glob(f"{outemp}/*.png")]
            frame_one = frames[0]
            frame_one.save(self.opt.Output_path.replace('png', 'gif'), format="GIF", append_images=frames,
                           save_all=True, duration=1000, loop=0)
            shutil.rmtree(outemp)

    def run(self):
        start, end = self.opt.read_start, self.opt.read_end
        GT, estimation = self.load_data(start, end)
        error = []
        for es in estimation:
            error.append(self.error(GT, es))
        self.draw_map(GT, error, estimation)


def main(opt):
    hm = heat_map(opt)
    hm.run()


if __name__ == '__main__':
    opt = options()
    main(opt)
