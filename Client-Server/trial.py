import argparse
import json
from tqdm import tqdm
from PIL import Image,ImageDraw
import numpy as np
import os
import math

def options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trials_path', type=str, help='trials path')
    parser.add_argument('--plan_path', type=str, help='floor plan path')
    parser.add_argument('--output', type=str, help='output path')
    opt = parser.parse_args()
    return opt

def star_vertices(plot_scale,center,r):
    out_vertex = [(r*plot_scale * np.cos(2 * np.pi * k / 5 + np.pi / 2- np.pi / 5) + center[0],
                   r*plot_scale * np.sin(2 * np.pi * k / 5 + np.pi / 2- np.pi / 5) + center[1]) for k in range(5)]
    r = r/2
    in_vertex = [(r*plot_scale * np.cos(2 * np.pi * k / 5 + np.pi / 2 ) + center[0],
                  r*plot_scale * np.sin(2 * np.pi * k / 5 + np.pi / 2 ) + center[1]) for k in range(5)]
    vertices = []
    for i in range(5):
        vertices.append(out_vertex[i])
        vertices.append(in_vertex[i])
    vertices = tuple(vertices)
    return vertices

def arrowedLine(draw, ptA, ptB, width=1, color=(0,255,0)):
    """Draw line from ptA to ptB with arrowhead at ptB"""
    # Get drawing context
    # Draw the line without arrows
    draw.line((ptA,ptB), width=width, fill=color)

    # Now work out the arrowhead
    # = it will be a triangle with one vertex at ptB
    # - it will start at 95% of the length of the line
    # - it will extend 8 pixels either side of the line
    x0, y0 = ptA
    x1, y1 = ptB
    # Now we can work out the x,y coordinates of the bottom of the arrowhead triangle
    xb = 0.95*(x1-x0)+x0
    yb = 0.95*(y1-y0)+y0

    # Work out the other two vertices of the triangle
    # Check if line is vertical
    if x0==x1:
       vtx0 = (xb-5, yb)
       vtx1 = (xb+5, yb)
    # Check if line is horizontal
    elif y0==y1:
       vtx0 = (xb, yb+5)
       vtx1 = (xb, yb-5)
    else:
       alpha = math.atan2(y1-y0,x1-x0)-90*math.pi/180
       a = 8*math.cos(alpha)
       b = 8*math.sin(alpha)
       vtx0 = (xb+a, yb+b)
       vtx1 = (xb-a, yb-b)

    #draw.point((xb,yb), fill=(255,0,0))    # DEBUG: draw point of base in red - comment out draw.polygon() below if using this line
    #im.save('DEBUG-base.png')              # DEBUG: save

    # Now draw the arrowhead triangle
    draw.polygon([vtx0, vtx1, ptB], fill=color)

def print_trial(data,floorplan_path,root):
    start_time=data['start time']
    stop_time=data['stop time']
    waypoints=data['waypoints']
    destination=data['destination']
    Place=destination[0]
    Building=destination[1]
    Floor=destination[2]
    outpath=os.path.join(root,Place,Building,Floor)
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    index=str(len(os.listdir(outpath))).zfill(5)
    outpath=os.path.join(outpath,index)
    os.mkdir(outpath)
    x_,y_=destination[3]
    floorplan=Image.open(os.path.join(floorplan_path,Place,Building,Floor,'floorplan.png'))
    draw = ImageDraw.Draw(floorplan)
    width, height = floorplan.size
    plot_scale = width / 3400
    vertices = star_vertices(plot_scale,[x_, y_], 30)
    draw.polygon(vertices, fill='green')
    x0, y0 = waypoints[0]
    vertices = star_vertices(plot_scale, [x0, y0], 15)
    draw.polygon(vertices, fill='green', outline='red')
    if len(waypoints)>1:
        for i in range(1,len(waypoints)):
            x1,y1=waypoints[i]
            vertices = star_vertices(plot_scale, [x1, y1], 15)
            draw.polygon(vertices, fill='black', outline='red')
            arrowedLine(draw,(x0, y0), (x1, y1), color=(255, 0, 0), width=5)
            # draw.line([(x0, y0), (x1, y1)], fill=(255, 0, 255), width=5)
            x0,y0=x1,y1
    floorplan.save(os.path.join(outpath,'trial.png'))

def main(opt):
    floorplan_path=opt.plan_path
    outf=opt.output
    with open(opt.trials_path,'r') as f:
        trials=json.load(f)
    for k,v in tqdm(trials.items()):
        print_trial(v,floorplan_path,outf)

if __name__ == '__main__':
    opt = options()
    main(opt)