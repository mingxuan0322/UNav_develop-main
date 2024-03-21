import tkinter as tk
from tkinter import *
import numpy as np
from tkinter import ttk,filedialog
from tkinter.messagebox import showinfo
from PIL import Image, ImageTk, ImageDraw, ImageOps
from tqdm import tqdm
import argparse
import json
import os
from os.path import join, exists, isfile
from ttkbootstrap import Style
import natsort
import cv2
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import h5py
import pytorch_NetVlad.netvlad as netvlad
from SuperPoint_SuperGlue.base_model import dynamic_load
from SuperPoint_SuperGlue import extractors
from SuperPoint_SuperGlue import matchers
import warnings
import math
from scipy.spatial.transform import Rotation as R
from scipy.spatial import KDTree
from sklearn.ensemble import IsolationForest
import pycolmap
from skimage.measure import ransac
from skimage.transform import AffineTransform
import pyimplicitdist
import poselib

def options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--topomap_path', default=None, required=True,
                        help='path to saved')
    parser.add_argument('--GT_path', default=None, required=True,
                        help='ground truth path')
    parser.add_argument('--path_path', default=None, required=True,
                        help='path to path')
    parser.add_argument('--fl_path', default=None, required=True,
                        help='path to floor plan')
    parser.add_argument('--global_descriptor', type=str, help='global descriptor path')
    parser.add_argument('--local_descriptor', type=str, help='local descriptor path')
    parser.add_argument('--db_dir', type=str, help='database image path')
    parser.add_argument('--query_dir', type=str, help='query image path')
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

def load_data(path):
    if os.path.exists(os.path.join(path, 'topo-map.json')):
        with open(os.path.join(path, 'topo-map.json'), 'r') as f:
            data = json.load(f)
    else:
        print('Topomap at '+path+'does not exists!')
        exit()
    return data

def camera_query():
    camera_model='SIMPLE_RADIAL'
    width=640
    height=360
    params=[5.52485707e+02,  3.20000000e+02,  1.80000000e+02, -7.24958617e-03]
    # params = [586.18491588,  586.3813968,  322.8904483, 186.26885421]
    # params = [1695.99583, 1697.25415, 958.536782, 533.007660]
    cfg = {
        'model': camera_model,
        'width': width,
        'height': height,
        'params': params,
    }
    return cfg

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

class AutoScrollbar(ttk.Scrollbar):
    """ A scrollbar that hides itself if it's not needed. Works only for grid geometry manager """
    def set(self, lo, hi):
        if float(lo) <= 0.0 and float(hi) >= 1.0:
            self.grid_remove()
        else:
            self.grid()
            ttk.Scrollbar.set(self, lo, hi)

    def pack(self, **kw):
        raise tk.TclError('Cannot use pack with the widget ' + self.__class__.__name__)

    def place(self, **kw):
        raise tk.TclError('Cannot use place with the widget ' + self.__class__.__name__)

class Button(ttk.Button):
    """ A scrollbar that hides itself if it's not needed. Works only for grid geometry manager """
    def set(self, lo, hi):
        if float(lo) <= 0.0 and float(hi) >= 1.0:
            self.grid_remove()
        else:
            self.grid()
            ttk.Scrollbar.set(self, lo, hi)

    def pack(self, **kw):
        raise tk.TclError('Cannot use pack with the widget ' + self.__class__.__name__)

    def place(self, **kw):
        raise tk.TclError('Cannot use place with the widget ' + self.__class__.__name__)

class CanvasImage_Pairs:
    """ Display and zoom image """
    def __init__(self, placeholder,image):
        """ Initialize the ImageFrame """
        self.placeholder=placeholder
        self.imscale = 1.0  # scale for the canvas image zoom, public for outer classes
        self.__delta = 1.1  # zoom magnitude
        self.__filter = Image.ANTIALIAS  # could be: NEAREST, BILINEAR, BICUBIC and ANTIALIAS
        self.__previous_state = 0  # previous state of the keyboard
        self.image = image  # path to the image, should be public for outer classes
        self.text=None
        # Create ImageFrame in placeholder widget
        self.__imframe = ttk.Frame(placeholder)  # placeholder of the ImageFrame object
        # Vertical and horizontal scrollbars for canvas
        hbar = AutoScrollbar(self.__imframe, orient='horizontal')
        vbar = AutoScrollbar(self.__imframe, orient='vertical')
        hbar.grid(row=1, column=0, sticky='we')
        vbar.grid(row=0, column=1, sticky='ns')
        # Create canvas and bind it with scrollbars. Public for outer classes
        self.canvas = tk.Canvas(self.__imframe, highlightthickness=0,
                                xscrollcommand=hbar.set, yscrollcommand=vbar.set)
        self.canvas.grid(row=0, column=0, sticky='nswe')
        self.canvas.update()  # wait till canvas is created
        hbar.configure(command=self.__scroll_x)  # bind scrollbars to the canvas
        vbar.configure(command=self.__scroll_y)
        # Bind events to the Canvas
        self.canvas.bind('<Configure>', lambda event: self.__show_image())  # canvas is resized
        self.canvas.bind('<Button-1>', self.__move_from)  # remember canvas position
        self.canvas.bind('<B1-Motion>',     self.__move_to)  # move canvas to the new position
        self.canvas.bind('<MouseWheel>', self.__wheel)  # zoom for Windows and MacOS, but not Linux
        self.canvas.bind('<Button-5>',   self.__wheel)  # zoom for Linux, wheel scroll down
        self.canvas.bind('<Button-4>',   self.__wheel)  # zoom for Linux, wheel scroll up
        # Handle keystrokes in idle mode, because program slows down on a weak computers,
        # when too many key stroke events in the same time
        self.canvas.bind('<Key>', lambda event: self.canvas.after_idle(self.__keystroke, event))
        # Decide if this image huge or not
        self.__huge = False  # huge or not
        self.__huge_size = 14000  # define size of the huge image
        self.__band_width = 1024  # width of the tile band
        Image.MAX_IMAGE_PIXELS = 1000000000  # suppress DecompressionBombError for the big image
        with warnings.catch_warnings():  # suppress DecompressionBombWarning
            warnings.simplefilter('ignore')
            self.__image = self.image # open image, but down't load it
        self.imwidth, self.imheight = self.__image.size  # public for outer classes
        if self.imwidth * self.imheight > self.__huge_size * self.__huge_size and \
           self.__image.tile[0][0] == 'raw':  # only raw images could be tiled
            self.__huge = True  # image is huge
            self.__offset = self.__image.tile[0][2]  # initial tile offset
            self.__tile = [self.__image.tile[0][0],  # it have to be 'raw'
                           [0, 0, self.imwidth, 0],  # tile extent (a rectangle)
                           self.__offset,
                           self.__image.tile[0][3]]  # list of arguments to the decoder
        self.__min_side = min(self.imwidth, self.imheight)  # get the smaller image side
        # Create image pyramid
        self.__pyramid = [self.smaller()] if self.__huge else [self.image]
        # Set ratio coefficient for image pyramid
        self.__ratio = max(self.imwidth, self.imheight) / self.__huge_size if self.__huge else 1.0
        self.__curr_img = 0  # current image from the pyramid
        self.__scale = self.imscale * self.__ratio  # image pyramide scale
        self.__reduction = 2  # reduction degree of image pyramid
        w, h = self.__pyramid[-1].size
        while w > 512 and h > 512:  # top pyramid image is around 512 pixels in size
            w /= self.__reduction  # divide on reduction degree
            h /= self.__reduction  # divide on reduction degree
            self.__pyramid.append(self.__pyramid[-1].resize((int(w), int(h)), self.__filter))
        # Put image into container rectangle and use it to set proper coordinates to the image
        self.container = self.canvas.create_rectangle((0, 0, self.imwidth, self.imheight), width=0)
        minsize, maxsize, number = 5, 20, 10
        self.width, self.height = self.__image.size

        self.__show_image()  # show image on the canvas
        self.canvas.focus_set()  # set focus on the canvas

    def smaller(self):
        """ Resize image proportionally and return smaller image """
        w1, h1 = float(self.imwidth), float(self.imheight)
        w2, h2 = float(self.__huge_size), float(self.__huge_size)
        aspect_ratio1 = w1 / h1
        aspect_ratio2 = w2 / h2  # it equals to 1.0
        if aspect_ratio1 == aspect_ratio2:
            image = Image.new('RGB', (int(w2), int(h2)))
            k = h2 / h1  # compression ratio
            w = int(w2)  # band length
        elif aspect_ratio1 > aspect_ratio2:
            image = Image.new('RGB', (int(w2), int(w2 / aspect_ratio1)))
            k = h2 / w1  # compression ratio
            w = int(w2)  # band length
        else:  # aspect_ratio1 < aspect_ration2
            image = Image.new('RGB', (int(h2 * aspect_ratio1), int(h2)))
            k = h2 / h1  # compression ratio
            w = int(h2 * aspect_ratio1)  # band length
        i, j, n = 0, 1, round(0.5 + self.imheight / self.__band_width)

        while i < self.imheight:
            print('\rOpening image: {j} from {n}'.format(j=j, n=n), end='')
            band = min(self.__band_width, self.imheight - i)  # width of the tile band
            self.__tile[1][3] = band  # set band width
            self.__tile[2] = self.__offset + self.imwidth * i * 3  # tile offset (3 bytes per pixel)
            self.__image.close()
            self.__image = image  # reopen / reset image
            self.__image.size = (self.imwidth, band)  # set size of the tile band
            self.__image.tile = [self.__tile]  # set tile
            cropped = self.__image.crop((0, 0, self.imwidth, band))  # crop tile band
            image.paste(cropped.resize((w, int(band * k)+1), self.__filter), (0, int(i * k)))
            i += band
            j += 1
        print('\r' + 30*' ' + '\r', end='')  # hide printed string
        return image

    def redraw_figures(self):
        """ Dummy function to redraw figures in the children classes """
        pass

    def grid(self, **kw):
        """ Put CanvasImage widget on the parent widget """
        self.__imframe.grid(**kw)  # place CanvasImage widget on the grid
        self.__imframe.grid(sticky='nswe')  # make frame container sticky
        self.__imframe.rowconfigure(0, weight=1)  # make canvas expandable
        self.__imframe.columnconfigure(0, weight=1)

    def pack(self, **kw):
        """ Exception: cannot use pack with this widget """
        raise Exception('Cannot use pack with the widget ' + self.__class__.__name__)

    def place(self, **kw):
        """ Exception: cannot use place with this widget """
        raise Exception('Cannot use place with the widget ' + self.__class__.__name__)

    # noinspection PyUnusedLocal
    def __scroll_x(self, *args, **kwargs):
        """ Scroll canvas horizontally and redraw the image """
        self.canvas.xview(*args)  # scroll horizontally
        self.__show_image()  # redraw the image

    # noinspection PyUnusedLocal
    def __scroll_y(self, *args, **kwargs):
        """ Scroll canvas vertically and redraw the image """
        self.canvas.yview(*args)  # scroll vertically
        self.__show_image()  # redraw the image

    def __show_image(self):
        """ Show image on the Canvas. Implements correct image zoom almost like in Google Maps """
        box_image = self.canvas.coords(self.container)  # get image area
        box_canvas = (self.canvas.canvasx(0),  # get visible area of the canvas
                      self.canvas.canvasy(0),
                      self.canvas.canvasx(self.canvas.winfo_width()),
                      self.canvas.canvasy(self.canvas.winfo_height()))
        box_img_int = tuple(map(int, box_image))  # convert to integer or it will not work properly
        # Get scroll region box
        box_scroll = [min(box_img_int[0], box_canvas[0]), min(box_img_int[1], box_canvas[1]),
                      max(box_img_int[2], box_canvas[2]), max(box_img_int[3], box_canvas[3])]
        # Horizontal part of the image is in the visible area
        if  box_scroll[0] == box_canvas[0] and box_scroll[2] == box_canvas[2]:
            box_scroll[0]  = box_img_int[0]
            box_scroll[2]  = box_img_int[2]
        # Vertical part of the image is in the visible area
        if  box_scroll[1] == box_canvas[1] and box_scroll[3] == box_canvas[3]:
            box_scroll[1]  = box_img_int[1]
            box_scroll[3]  = box_img_int[3]
        # Convert scroll region to tuple and to integer
        self.canvas.configure(scrollregion=tuple(map(int, box_scroll)))  # set scroll region
        x1 = max(box_canvas[0] - box_image[0], 0)  # get coordinates (x1,y1,x2,y2) of the image tile
        y1 = max(box_canvas[1] - box_image[1], 0)
        x2 = min(box_canvas[2], box_image[2]) - box_image[0]
        y2 = min(box_canvas[3], box_image[3]) - box_image[1]
        if int(x2 - x1) > 0 and int(y2 - y1) > 0:  # show image if it in the visible area
            if self.__huge and self.__curr_img < 0:  # show huge image
                h = int((y2 - y1) / self.imscale)  # height of the tile band
                self.__tile[1][3] = h  # set the tile band height
                self.__tile[2] = self.__offset + self.imwidth * int(y1 / self.imscale) * 3
                self.__image.close()
                self.__image = self.image  # reopen / reset image
                self.__image.size = (self.imwidth, h)  # set size of the tile band
                self.__image.tile = [self.__tile]
                image = self.__image.crop((int(x1 / self.imscale), 0, int(x2 / self.imscale), h))
            else:  # show normal image
                image = self.__pyramid[max(0, self.__curr_img)].crop(  # crop current img from pyramid
                                    (int(x1 / self.__scale), int(y1 / self.__scale),
                                     int(x2 / self.__scale), int(y2 / self.__scale)))
            #
            imagetk = ImageTk.PhotoImage(image.resize((int(x2 - x1), int(y2 - y1)), self.__filter))
            imageid = self.canvas.create_image(max(box_canvas[0], box_img_int[0]),
                                               max(box_canvas[1], box_img_int[1]),
                                               anchor='nw', image=imagetk)
            self.canvas.lower(imageid)  # set image into background
            self.canvas.imagetk = imagetk  # keep an extra reference to prevent garbage-collection

    def __move_from(self, event):
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        bbox = self.canvas.coords(self.container)
        scale= (bbox[2] - bbox[0]) / self.imwidth
        """ Remember previous coordinates for scrolling with the mouse """
        self.canvas.scan_mark(event.x, event.y)

    def __move_to(self, event):
        """ Drag (move) canvas to the new position """
        self.canvas.scan_dragto(event.x, event.y, gain=1)
        self.__show_image()  # zoom tile and show it on the canvas

    def outside(self, x, y):
        """ Checks if the point (x,y) is outside the image area """
        bbox = self.canvas.coords(self.container)  # get image area
        if bbox[0] < x < bbox[2] and bbox[1] < y < bbox[3]:
            return False  # point (x,y) is inside the image area
        else:
            return True  # point (x,y) is outside the image area

    def __wheel(self, event):
        """ Zoom with mouse wheel """
        x = self.canvas.canvasx(event.x)  # get coordinates of the event on the canvas
        y = self.canvas.canvasy(event.y)
        if self.outside(x, y): return  # zoom only inside image area
        scale = 1.0
        # Respond to Linux (event.num) or Windows (event.delta) wheel event
        if event.num == 5 or event.delta == -120:  # scroll down, smaller
            if round(self.__min_side * self.imscale) < 30: return  # image is less than 30 pixels
            self.imscale /= self.__delta
            scale        /= self.__delta
        if event.num == 4 or event.delta == 120:  # scroll up, bigger
            i = min(self.canvas.winfo_width(), self.canvas.winfo_height()) >> 1
            if i < self.imscale: return  # 1 pixel is bigger than the visible area
            self.imscale *= self.__delta
            scale        *= self.__delta
        # Take appropriate image from the pyramid
        k = self.imscale * self.__ratio  # temporary coefficient
        self.__curr_img = min((-1) * int(math.log(k, self.__reduction)), len(self.__pyramid) - 1)
        self.__scale = k * math.pow(self.__reduction, max(0, self.__curr_img))
        #
        self.canvas.scale('all', x, y, scale, scale)  # rescale all objects
        # Redraw some figures before showing image on the screen
        self.redraw_figures()  # method for child classes
        self.__show_image()

    def __keystroke(self, event):
        """ Scrolling with the keyboard.
            Independent from the language of the keyboard, CapsLock, <Ctrl>+<key>, etc. """
        if event.state - self.__previous_state == 4:  # means that the Control key is pressed
            pass  # do nothing if Control key is pressed
        else:
            self.__previous_state = event.state  # remember the last keystroke state
            # Up, Down, Left, Right keystrokes
            if event.keycode in [68, 39, 102]:  # scroll right: keys 'D', 'Right' or 'Numpad-6'
                self.__scroll_x('scroll',  1, 'unit', event=event)
            elif event.keycode in [65, 37, 100]:  # scroll left: keys 'A', 'Left' or 'Numpad-4'
                self.__scroll_x('scroll', -1, 'unit', event=event)
            elif event.keycode in [87, 38, 104]:  # scroll up: keys 'W', 'Up' or 'Numpad-8'
                self.__scroll_y('scroll', -1, 'unit', event=event)
            elif event.keycode in [83, 40, 98]:  # scroll down: keys 'S', 'Down' or 'Numpad-2'
                self.__scroll_y('scroll',  1, 'unit', event=event)

class CanvasImage_Trajectory:
    """ Display and zoom image """
    def __init__(self, placeholder,parent):
        """ Initialize the ImageFrame """
        self.parent=parent
        self.placeholder=placeholder
        self.imscale = 1.0  # scale for the canvas image zoom, public for outer classes
        self.__delta = 1.1  # zoom magnitude
        self.__filter = Image.ANTIALIAS  # could be: NEAREST, BILINEAR, BICUBIC and ANTIALIAS
        self.__previous_state = 0  # previous state of the keyboard
        self.text=None
        # Create ImageFrame in placeholder widget
        self.__imframe = ttk.Frame(placeholder)  # placeholder of the ImageFrame object
        # Vertical and horizontal scrollbars for canvas
        hbar = AutoScrollbar(self.__imframe, orient='horizontal')
        vbar = AutoScrollbar(self.__imframe, orient='vertical')
        hbar.grid(row=1, column=0, sticky='we')
        vbar.grid(row=0, column=1, sticky='ns')
        # Create canvas and bind it with scrollbars. Public for outer classes
        self.canvas = tk.Canvas(self.__imframe, highlightthickness=0,
                                xscrollcommand=hbar.set, yscrollcommand=vbar.set)
        self.canvas.grid(row=0, column=0, sticky='nswe')
        self.canvas.update()  # wait till canvas is created
        hbar.configure(command=self.__scroll_x)  # bind scrollbars to the canvas
        vbar.configure(command=self.__scroll_y)
        # Bind events to the Canvas
        self.canvas.bind('<Configure>', lambda event: self.__show_image())  # canvas is resized
        self.canvas.bind('<Button-1>', self.__move_from)  # remember canvas position
        self.canvas.bind('<Button-3>', self.__coordinates)
        self.canvas.bind('<B1-Motion>',     self.__move_to)  # move canvas to the new position
        self.canvas.bind('<MouseWheel>', self.__wheel)  # zoom for Windows and MacOS, but not Linux
        self.canvas.bind('<Button-5>',   self.__wheel)  # zoom for Linux, wheel scroll down
        self.canvas.bind('<Button-4>',   self.__wheel)  # zoom for Linux, wheel scroll up
        # Handle keystrokes in idle mode, because program slows down on a weak computers,
        # when too many key stroke events in the same time
        self.canvas.bind('<Key>', lambda event: self.canvas.after_idle(self.__keystroke, event))
        # Decide if this image huge or not
        self.__huge = False  # huge or not
        self.__huge_size = 14000  # define size of the huge image
        self.__band_width = 1024  # width of the tile band
        Image.MAX_IMAGE_PIXELS = 1000000000  # suppress DecompressionBombError for the big image
        with warnings.catch_warnings():  # suppress DecompressionBombWarning
            warnings.simplefilter('ignore')
            self.__image = self.parent.imtra  # open image, but down't load it
        self.imwidth, self.imheight = self.__image.size  # public for outer classes
        if self.imwidth * self.imheight > self.__huge_size * self.__huge_size and \
           self.__image.tile[0][0] == 'raw':  # only raw images could be tiled
            self.__huge = True  # image is huge
            self.__offset = self.__image.tile[0][2]  # initial tile offset
            self.__tile = [self.__image.tile[0][0],  # it have to be 'raw'
                           [0, 0, self.imwidth, 0],  # tile extent (a rectangle)
                           self.__offset,
                           self.__image.tile[0][3]]  # list of arguments to the decoder
        self.__min_side = min(self.imwidth, self.imheight)  # get the smaller image side
        # Create image pyramid
        self.__pyramid = [self.smaller()] if self.__huge else [self.parent.imtra]
        # Set ratio coefficient for image pyramid
        self.__ratio = max(self.imwidth, self.imheight) / self.__huge_size if self.__huge else 1.0
        self.__curr_img = 0  # current image from the pyramid
        self.__scale = self.imscale * self.__ratio  # image pyramide scale
        self.__reduction = 2  # reduction degree of image pyramid
        w, h = self.__pyramid[-1].size
        while w > 512 and h > 512:  # top pyramid image is around 512 pixels in size
            w /= self.__reduction  # divide on reduction degree
            h /= self.__reduction  # divide on reduction degree
            self.__pyramid.append(self.__pyramid[-1].resize((int(w), int(h)), self.__filter))
        # Put image into container rectangle and use it to set proper coordinates to the image
        self.container = self.canvas.create_rectangle((0, 0, self.imwidth, self.imheight), width=0)
        minsize, maxsize, number = 5, 20, 10
        self.width, self.height = self.__image.size
        self.r=3
        self.chose={}
        self.nochose={}
        self.picked=[]

        for i in range(len(self.parent.knames)):
            x0, y0 = self.parent.pts[i]
            self.nochose.update({str(x0) + '-' + str(y0): {
                'index': self.canvas.create_oval(x0 - self.r, y0 - self.r, x0 + self.r, y0 + self.r, fill='green',
                                                 activefill='red'), 'id': self.parent.knames[i]}})
        # self.index=list(self.parent.data['keyframes'].keys())
        # for index in self.index:
        #     x0,y0=self.parent.data['keyframes'][index]['trans']
        #     self.nochose.update({str(x0)+'-'+str(y0):{'index':self.canvas.create_oval(x0-self.r, y0-self.r, x0+self.r, y0+self.r, fill='green', activefill='red'),'id':index}})
        self.__show_image()  # show image on the canvas
        self.canvas.focus_set()  # set focus on the canvas

    def __coordinates(self,event):
        if len(self.chose)==1:
            x=self.canvas.canvasx(event.x)
            y=self.canvas.canvasy(event.y)
            bbox = self.canvas.coords(self.container)
            scale = (bbox[2] - bbox[0]) / self.imwidth
            x_, y_ = (x - bbox[0]) / scale, (y - bbox[1]) / scale
            for s in self.chose.keys():
                list1=s.split('-')
                x0,y0=float(list1[0]),float(list1[1])
                if (np.linalg.norm((x_ - x0, y_ - y0)) < self.r * 2):
                    self.parent.destination.append(self.chose[s]['id'])
                    self.parent.set_destination()
                    self.placeholder.destroy()

    def smaller(self):
        """ Resize image proportionally and return smaller image """
        w1, h1 = float(self.imwidth), float(self.imheight)
        w2, h2 = float(self.__huge_size), float(self.__huge_size)
        aspect_ratio1 = w1 / h1
        aspect_ratio2 = w2 / h2  # it equals to 1.0
        if aspect_ratio1 == aspect_ratio2:
            image = Image.new('RGB', (int(w2), int(h2)))
            k = h2 / h1  # compression ratio
            w = int(w2)  # band length
        elif aspect_ratio1 > aspect_ratio2:
            image = Image.new('RGB', (int(w2), int(w2 / aspect_ratio1)))
            k = h2 / w1  # compression ratio
            w = int(w2)  # band length
        else:  # aspect_ratio1 < aspect_ration2
            image = Image.new('RGB', (int(h2 * aspect_ratio1), int(h2)))
            k = h2 / h1  # compression ratio
            w = int(h2 * aspect_ratio1)  # band length
        i, j, n = 0, 1, round(0.5 + self.imheight / self.__band_width)

        while i < self.imheight:
            print('\rOpening image: {j} from {n}'.format(j=j, n=n), end='')
            band = min(self.__band_width, self.imheight - i)  # width of the tile band
            self.__tile[1][3] = band  # set band width
            self.__tile[2] = self.__offset + self.imwidth * i * 3  # tile offset (3 bytes per pixel)
            self.__image.close()
            self.__image = Image.open(self.path)  # reopen / reset image
            self.__image.size = (self.imwidth, band)  # set size of the tile band
            self.__image.tile = [self.__tile]  # set tile
            cropped = self.__image.crop((0, 0, self.imwidth, band))  # crop tile band
            image.paste(cropped.resize((w, int(band * k)+1), self.__filter), (0, int(i * k)))
            i += band
            j += 1
        print('\r' + 30*' ' + '\r', end='')  # hide printed string
        return image

    def redraw_figures(self):
        """ Dummy function to redraw figures in the children classes """
        pass

    def grid(self, **kw):
        """ Put CanvasImage widget on the parent widget """
        self.__imframe.grid(**kw)  # place CanvasImage widget on the grid
        self.__imframe.grid(sticky='nswe')  # make frame container sticky
        self.__imframe.rowconfigure(0, weight=1)  # make canvas expandable
        self.__imframe.columnconfigure(0, weight=1)

    def pack(self, **kw):
        """ Exception: cannot use pack with this widget """
        raise Exception('Cannot use pack with the widget ' + self.__class__.__name__)

    def place(self, **kw):
        """ Exception: cannot use place with this widget """
        raise Exception('Cannot use place with the widget ' + self.__class__.__name__)

    # noinspection PyUnusedLocal
    def __scroll_x(self, *args, **kwargs):
        """ Scroll canvas horizontally and redraw the image """
        self.canvas.xview(*args)  # scroll horizontally
        self.__show_image()  # redraw the image

    # noinspection PyUnusedLocal
    def __scroll_y(self, *args, **kwargs):
        """ Scroll canvas vertically and redraw the image """
        self.canvas.yview(*args)  # scroll vertically
        self.__show_image()  # redraw the image

    def __show_image(self):
        """ Show image on the Canvas. Implements correct image zoom almost like in Google Maps """
        box_image = self.canvas.coords(self.container)  # get image area
        box_canvas = (self.canvas.canvasx(0),  # get visible area of the canvas
                      self.canvas.canvasy(0),
                      self.canvas.canvasx(self.canvas.winfo_width()),
                      self.canvas.canvasy(self.canvas.winfo_height()))
        box_img_int = tuple(map(int, box_image))  # convert to integer or it will not work properly
        # Get scroll region box
        box_scroll = [min(box_img_int[0], box_canvas[0]), min(box_img_int[1], box_canvas[1]),
                      max(box_img_int[2], box_canvas[2]), max(box_img_int[3], box_canvas[3])]
        # Horizontal part of the image is in the visible area
        if  box_scroll[0] == box_canvas[0] and box_scroll[2] == box_canvas[2]:
            box_scroll[0]  = box_img_int[0]
            box_scroll[2]  = box_img_int[2]
        # Vertical part of the image is in the visible area
        if  box_scroll[1] == box_canvas[1] and box_scroll[3] == box_canvas[3]:
            box_scroll[1]  = box_img_int[1]
            box_scroll[3]  = box_img_int[3]
        # Convert scroll region to tuple and to integer
        self.canvas.configure(scrollregion=tuple(map(int, box_scroll)))  # set scroll region
        x1 = max(box_canvas[0] - box_image[0], 0)  # get coordinates (x1,y1,x2,y2) of the image tile
        y1 = max(box_canvas[1] - box_image[1], 0)
        x2 = min(box_canvas[2], box_image[2]) - box_image[0]
        y2 = min(box_canvas[3], box_image[3]) - box_image[1]
        if int(x2 - x1) > 0 and int(y2 - y1) > 0:  # show image if it in the visible area
            if self.__huge and self.__curr_img < 0:  # show huge image
                h = int((y2 - y1) / self.imscale)  # height of the tile band
                self.__tile[1][3] = h  # set the tile band height
                self.__tile[2] = self.__offset + self.imwidth * int(y1 / self.imscale) * 3
                self.__image.close()
                self.__image = Image.open(self.path)  # reopen / reset image
                self.__image.size = (self.imwidth, h)  # set size of the tile band
                self.__image.tile = [self.__tile]
                image = self.__image.crop((int(x1 / self.imscale), 0, int(x2 / self.imscale), h))
            else:  # show normal image
                image = self.__pyramid[max(0, self.__curr_img)].crop(  # crop current img from pyramid
                                    (int(x1 / self.__scale), int(y1 / self.__scale),
                                     int(x2 / self.__scale), int(y2 / self.__scale)))
            #
            imagetk = ImageTk.PhotoImage(image.resize((int(x2 - x1), int(y2 - y1)), self.__filter))
            imageid = self.canvas.create_image(max(box_canvas[0], box_img_int[0]),
                                               max(box_canvas[1], box_img_int[1]),
                                               anchor='nw', image=imagetk)
            self.canvas.lower(imageid)  # set image into background
            self.canvas.imagetk = imagetk  # keep an extra reference to prevent garbage-collection

    def __move_from(self, event):
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        bbox = self.canvas.coords(self.container)
        scale= (bbox[2] - bbox[0]) / self.imwidth
        x_, y_ = (x - bbox[0]) / scale, (y - bbox[1]) / scale
        for i in range(len(self.parent.knames)):
            x0, y0 = self.parent.pts[i]
            if (np.linalg.norm((x_-x0,y_-y0))<self.r*2)and(str(x0)+'-'+str(y0) in self.chose):
                x0l, x0r, y0l, y0r = (x0 - self.r) * scale + bbox[0], (x0 + self.r) * scale + bbox[0], (
                            y0 - self.r) * scale + bbox[1], (y0 + self.r) * scale + bbox[1]
                self.nochose.update({str(x0) + '-' + str(y0):{'index':self.canvas.create_oval(x0l,y0l,x0r,y0r, fill='green', activefill='red'),'id':self.parent.knames[i]}})
                self.canvas.delete(self.chose[str(x0) + '-' + str(y0)]['index'])
                self.chose.pop(str(x0)+'-'+str(y0))
                break
            if (np.linalg.norm((x_ - x0, y_ - y0)) < self.r) and (str(x0) + '-' + str(y0) in self.nochose):
                if self.text:
                    self.canvas.delete(self.text)
                x0l, x0r, y0l, y0r = (x0 - self.r*2) * scale + bbox[0], (x0 + self.r*2) * scale + bbox[0], (
                            y0 - self.r*2) * scale + bbox[1], (y0 + self.r*2) * scale + bbox[1]
                self.text = self.canvas.create_text(x0* scale+ bbox[0], (y0 + self.r*3) * scale + bbox[1], fill="darkblue", font="Times 10 italic bold",
                                                    text="Right click to pick")
                self.chose.update({str(x0) + '-' + str(y0):{'index':self.canvas.create_oval(x0l,y0l,x0r,y0r, fill='red', activefill='green'),'id':self.parent.knames[i]}})
                self.canvas.delete(self.nochose[str(x0) + '-' + str(y0)]['index'])
                self.nochose.pop(str(x0) + '-' + str(y0))
                x_,y_=x0,y0
                if len(self.chose)>1:
                    for i in list(self.chose.keys())[:-1]:
                        x0,y0=i.split('-')
                        x0,y0=float(x0),float(y0)
                        x0l, x0r, y0l, y0r = (x0 - self.r) * scale + bbox[0], (x0 + self.r) * scale + bbox[0], (
                                y0 - self.r) * scale + bbox[1], (y0 + self.r) * scale + bbox[1]
                        self.nochose.update({i: {
                            'index': self.canvas.create_oval(x0l, y0l, x0r, y0r, fill='green', activefill='red'),'id':self.chose[i]['id']}})
                        self.canvas.delete(self.chose[i]['index'])
                        self.chose.pop(i)
                break
        """ Remember previous coordinates for scrolling with the mouse """
        self.canvas.scan_mark(event.x, event.y)

    def __move_to(self, event):
        """ Drag (move) canvas to the new position """
        self.canvas.scan_dragto(event.x, event.y, gain=1)
        self.__show_image()  # zoom tile and show it on the canvas

    def outside(self, x, y):
        """ Checks if the point (x,y) is outside the image area """
        bbox = self.canvas.coords(self.container)  # get image area
        if bbox[0] < x < bbox[2] and bbox[1] < y < bbox[3]:
            return False  # point (x,y) is inside the image area
        else:
            return True  # point (x,y) is outside the image area

    def __wheel(self, event):
        """ Zoom with mouse wheel """
        x = self.canvas.canvasx(event.x)  # get coordinates of the event on the canvas
        y = self.canvas.canvasy(event.y)
        if self.outside(x, y): return  # zoom only inside image area
        scale = 1.0
        # Respond to Linux (event.num) or Windows (event.delta) wheel event
        if event.num == 5 or event.delta == -120:  # scroll down, smaller
            if round(self.__min_side * self.imscale) < 30: return  # image is less than 30 pixels
            self.imscale /= self.__delta
            scale        /= self.__delta
        if event.num == 4 or event.delta == 120:  # scroll up, bigger
            i = min(self.canvas.winfo_width(), self.canvas.winfo_height()) >> 1
            if i < self.imscale: return  # 1 pixel is bigger than the visible area
            self.imscale *= self.__delta
            scale        *= self.__delta
        # Take appropriate image from the pyramid
        k = self.imscale * self.__ratio  # temporary coefficient
        self.__curr_img = min((-1) * int(math.log(k, self.__reduction)), len(self.__pyramid) - 1)
        self.__scale = k * math.pow(self.__reduction, max(0, self.__curr_img))
        #
        self.canvas.scale('all', x, y, scale, scale)  # rescale all objects
        # Redraw some figures before showing image on the screen
        self.redraw_figures()  # method for child classes
        self.__show_image()

    def __keystroke(self, event):
        """ Scrolling with the keyboard.
            Independent from the language of the keyboard, CapsLock, <Ctrl>+<key>, etc. """
        if event.state - self.__previous_state == 4:  # means that the Control key is pressed
            pass  # do nothing if Control key is pressed
        else:
            self.__previous_state = event.state  # remember the last keystroke state
            # Up, Down, Left, Right keystrokes
            if event.keycode in [68, 39, 102]:  # scroll right: keys 'D', 'Right' or 'Numpad-6'
                self.__scroll_x('scroll',  1, 'unit', event=event)
            elif event.keycode in [65, 37, 100]:  # scroll left: keys 'A', 'Left' or 'Numpad-4'
                self.__scroll_x('scroll', -1, 'unit', event=event)
            elif event.keycode in [87, 38, 104]:  # scroll up: keys 'W', 'Up' or 'Numpad-8'
                self.__scroll_y('scroll', -1, 'unit', event=event)
            elif event.keycode in [83, 40, 98]:  # scroll down: keys 'S', 'Down' or 'Numpad-2'
                self.__scroll_y('scroll',  1, 'unit', event=event)

class Pairs_window(ttk.Frame):
    def __init__(self,mainframe,pairs_image):
        ttk.Frame.__init__(self, master=mainframe)
        self.master.title('Pairs Matched')
        self.master.geometry('1200x600')  # size of the main window
        self.master.rowconfigure(0, weight=1)  # make the CanvasImage widget expandable
        self.master.columnconfigure(0, weight=1)

        canvas = CanvasImage_Pairs(self.master,pairs_image)  # create widget
        canvas.grid(row=0, column=0)  # show widget

class Trajectory_window(ttk.Frame):
    def __init__(self,mainframe,parent):
        ttk.Frame.__init__(self, master=mainframe)
        self.master.title('Pick a destination')
        self.master.geometry('1200x600')  # size of the main window
        self.master.rowconfigure(0, weight=1)  # make the CanvasImage widget expandable
        self.master.columnconfigure(0, weight=1)

        canvas = CanvasImage_Trajectory(self.master,parent)  # create widget
        canvas.grid(row=0, column=0)  # show widget

class Main_window(ttk.Frame):
    def __init__(self, master, data, GT_trajectory, opt):
        ttk.Frame.__init__(self, master=master)
        # self.temp_image=None
        self.colors=[(227,207,87),(138,43,226),(255,97,3),(102,205,0),(220,20,60),(178,58,238),(218,165,32),(127,255,212),(218,165,32)]
        self.camera_model=camera_query()
        self.ransac_threshold=12.0
        self.opt = opt
        self.data = data
        self.GT = GT_trajectory
        self.landmarks=self.data['landmarks']
        # self.temp
        self.keyframes = data['keyframes']
        self.pts = np.array([self.keyframes[k]['trans'] for k in list(self.keyframes.keys()) if k.split('_')[-1] == '00'], dtype=int)
        self.knames= [k.split('_')[0] for k in list(self.keyframes.keys()) if k.split('_')[-1] == '00']
        self.list_2d, self.list_3d, self.initial_poses, self.pps=[],[],[],[]

        self.kdtree = KDTree(self.pts)
        self.T=np.array(self.data['T'])
        self.rot_base = np.arctan2(self.T[1, 0], self.T[0, 0])
        self.globs = ['.jpg', '.png', '.jpeg', '.JPG', '.PNG']

        if os.path.exists(self.opt.path_path):
            self.path_file=h5py.File(self.opt.path_path,'r')['Path']
        else:
            self.path_file=None
        self.destination=[]

        self.hfile = h5py.File(self.opt.global_descriptor, 'r')
        self.hfile_local = h5py.File(self.opt.local_descriptor, 'r')

        names = []
        self.hfile.visititems(
            lambda _, obj: names.append(obj.parent.name.strip('/'))
            if isinstance(obj, h5py.Dataset) else None)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        kf_name = natsort.natsorted(list(set(self.data['keyframes'].keys())))

        self.n=len(list(set([n.split('_')[-1] for n in kf_name])))
        r = 30
        self.gamma = 2 * np.pi / self.n
        self.theta = np.pi / 2 - np.pi / self.n
        self.rbar = r * np.sin(self.theta)
        self.rhat = r * np.cos(self.theta)
        self.db_names = [n for n in names if n.replace('.png','') in kf_name]

        self.db_desc = self.tensor_from_names(self.db_names, self.hfile)

        # self.db_local=
        self.extractor = NetVladFeatureExtractor(opt.ckpt_path, arch=opt.arch, num_clusters=opt.num_clusters,
                                                 pooling=opt.pooling, vladv2=opt.vladv2, nocuda=opt.nocuda)
        self.conf = {
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
        self.conf_match = {
            'output': 'matches-superglue',
            'model': {
                'name': 'superglue',
                'weights': 'outdoor',
                'sinkhorn_iterations': 50,
            },
        }

        Model_sp = dynamic_load(extractors, self.conf['model']['name'])
        self.sp_model = Model_sp(self.conf['model']).eval().to(self.device)
        Model_sg = dynamic_load(matchers, self.conf_match['model']['name'])
        self.sg_model = Model_sg(self.conf_match['model']).eval().to(self.device)

        windowWidth = self.master.winfo_reqwidth()
        windowHeight = self.master.winfo_reqheight()
        self.positionRight = int(self.master.winfo_screenwidth() / 2 - windowWidth / 2)
        self.positionDown = int(self.master.winfo_screenheight() / 2 - windowHeight / 2)
        self.master.geometry("+{}+{}".format(self.positionRight, self.positionDown))
        self.master.title('Localization')
        self.pack(side="left", fill="both", expand=False)
        self.master.geometry('2000x1200')
        self.master.columnconfigure(1, weight=1)
        self.master.columnconfigure(3, pad=7)
        self.master.rowconfigure(3, weight=1)
        self.master.rowconfigure(6, pad=7)
        # --------------------------------------------------
        lbl = Label(self, text="Query frames:")
        lbl.grid(row=0, column=0, sticky=W, pady=4, ipadx=2)
        self.query_names=os.listdir(opt.query_dir)
        self.set_query(self.query_names)
        self.c=ttk.LabelFrame(self, text='Retrieval animation')
        self.c.grid(row=6, column=0, columnspan=2, padx=2,sticky=E + W)
        self.v1 = tk.IntVar()
        self.retrieval=True
        self.lb2 = tk.Radiobutton(self.c,
                                  text='Trun on',
                                  command=self.retrieval_setting,
                                  variable=self.v1,
                                  value=0).pack(side=LEFT)
        self.lb2 = tk.Radiobutton(self.c,
                                  text='Trun off',
                                  command=self.retrieval_setting,
                                  variable=self.v1,
                                  value=1).pack(side=RIGHT)
        self.shift = tk.Frame(self)
        self.shift.grid(row=14, column=0, padx=10, pady=1, sticky='we')
        scale = Label(self.shift, text='Retrieval number:')
        scale.pack(side=LEFT)
        self.e1 = tk.Entry(self.shift, width=3, justify='left')
        self.e1.pack(side=LEFT)
        self.e1.insert(END, '10')
        self.shift1 = tk.Frame(self)
        self.shift1.grid(row=12, column=0, padx=10, pady=1, sticky='we')
        scale = Label(self.shift1, text='Plotting scale:')
        scale.pack(side=LEFT)
        self.e2 = tk.Entry(self.shift1, width=5, justify='right')
        self.e2.insert(END, '0.1')
        self.e2.pack(side=LEFT)
        scale = Label(self.shift1, text='\'/pixel')
        scale.pack(side=LEFT)
        # ---------------------------------------------------
        separatorv1 = ttk.Separator(self, orient='vertical')
        separatorv1.grid(row=0, column=3, padx=10, columnspan=1, rowspan=70, sticky="sn")
        ebtn = tk.Button(self, text='Save Animation', width=16, command=self.gif_generator)
        ebtn.grid(row=15, column=0, padx=10, columnspan=1, rowspan=1)
        ebtn = tk.Button(self, text='Help', width=16, command=self.help)
        ebtn.grid(row=16, column=0, padx=10, columnspan=1, rowspan=1)
        ebtn = tk.Button(self, text='Clear Destination', width=16, command=self.clear_destination)
        ebtn.grid(row=17, column=0, padx=10, columnspan=1, rowspan=1)
        try:
            im=Image.open(self.data['floorplan'])
        except:
            fl_selected = filedialog.askopenfilename(initialdir=opt.fl_path, title='Select Floor Plan')
            self.data['floorplan']=fl_selected
            im = Image.open(fl_selected)
        draw = ImageDraw.Draw(im)
        self.imtra=im.copy()
        self.kf = self.data['keyframes']
        for index in self.kf.keys():
            k = self.kf[index]
            x_, y_ = k['trans']
            draw.ellipse((x_ - 2, y_ - 2, x_ + 2, y_ + 2), fill=(0, 255, 0))
        self.imf = im.copy()
        width, height = im.size
        self.plot_scale =width/3400
        scale = 1600 / width
        newsize = (1600, int(height * scale))
        im = im.resize(newsize)
        tkimage1 = ImageTk.PhotoImage(im)
        self.myvar1 = Label(self, image=tkimage1)
        self.myvar1.image = tkimage1
        self.myvar1.grid(row=0, column=4, columnspan=1, rowspan=40, sticky="snew")
        self.myvar1.bind('<Double-Button-1>', lambda event, action='double':
                            self.show_trajectory(action))

    def clear_destination(self):
        self.destination=[]
        self.set_destination()

    def set_query(self,dictionary):
        var2 = tk.StringVar()
        self.lb = tk.Listbox(self, listvariable=var2)

        self.scrollbar = Scrollbar(self, orient=VERTICAL)
        self.lb.config(yscrollcommand=self.scrollbar.set)
        self.scrollbar.config(command=self.lb.yview)

        self.lb.bind('<Double-Button-1>', lambda event, action='double':
        self.FloorPlan_select(action))
        self.lb.bind('<Up>', lambda event, action='up':
        self.FloorPlan_select(action))
        self.lb.bind('<Down>', lambda event, action='down':
        self.FloorPlan_select(action))
        self.dics = [i.split('.')[0] for i in sorted(dictionary)]
        for i in self.dics:
            self.lb.insert('end', i)

        self.scrollbar.grid(row=7, column=1, columnspan=2, rowspan=4, padx=2, sticky='sn')
        self.lb.grid(row=7, column=0, columnspan=1, rowspan=4, padx=2,
                     sticky=E + W + S + N)

    def retrieval_setting(self):
        if self.v1.get()==0:
            self.retrieval=True
        else:
            self.retrieval=False

    def prepare_data(self, image):
        image = np.array(ImageOps.grayscale(image)).astype(np.float32)
        image = image[None]
        data = torch.from_numpy(image / 255.).unsqueeze(0)
        return data

    def star_vertices(self,center,r):
        out_vertex = [(r*self.plot_scale * np.cos(2 * np.pi * k / 5 + np.pi / 2- np.pi / 5) + center[0],
                       r*self.plot_scale * np.sin(2 * np.pi * k / 5 + np.pi / 2- np.pi / 5) + center[1]) for k in range(5)]
        r = r/2
        in_vertex = [(r*self.plot_scale * np.cos(2 * np.pi * k / 5 + np.pi / 2 ) + center[0],
                      r*self.plot_scale * np.sin(2 * np.pi * k / 5 + np.pi / 2 ) + center[1]) for k in range(5)]
        vertices = []
        for i in range(5):
            vertices.append(out_vertex[i])
            vertices.append(in_vertex[i])
        vertices = tuple(vertices)
        return vertices

    def set_destination(self):
        im=self.imtra.copy()
        draw = ImageDraw.Draw(im)
        for i in range(len(self.knames)):
            if self.knames[i] not in self.destination:
                x_, y_ = self.pts[i]
                draw.ellipse((x_ - 2*self.plot_scale, y_ - 2*self.plot_scale, x_ + 2*self.plot_scale, y_ + 2*self.plot_scale), fill=(0, 255, 0))
        for d in self.destination:
            x_, y_ =self.pts[self.knames.index(d)]
            vertices = self.star_vertices([x_, y_],30)
            draw.polygon(vertices, fill='red')
        self.imf = im.copy()

        width, height = im.size
        scale = 1600 / width
        newsize = (1600, int(height * scale))
        im = im.resize(newsize)
        tkimage1 = ImageTk.PhotoImage(im)
        self.myvar1 = Label(self, image=tkimage1)
        self.myvar1.image = tkimage1
        self.myvar1.grid(row=0, column=4, columnspan=1, rowspan=40, sticky="snew")
        self.myvar1.bind('<Double-Button-1>', lambda event, action='double':
        self.show_trajectory(action))

    def get_start(self,Pr,x,y):
        _, i_ = self.kdtree.query((x,y),k=10)
        min_ang=np.inf
        i=None
        for index in i_:
            x0,y0=self.pts[index]
            if Pr[index]!=-9999:
                x1,y1=self.pts[Pr[index]]
                rot0 = np.arctan2(x - x0, y - y0)
                rot1 = np.arctan2(x0 - x1, y0 - y1)
                error=abs(rot0-rot1)
                if min_ang>error:
                    min_ang=error
                    i=index
        return i

    def get_path(self,Pr, j):
        try:
            paths = [self.knames[j]]
            k = j
            while Pr[k] != -9999:
                paths.append(self.knames[Pr[k]])
                k = Pr[k]
        except:
            paths=[-9999]
        return paths,k

    def cloest_path(self,x,y):
        indexs = self.get_start(self.path_file[self.destination[0]], x[0], y[0])
        paths=[]
        for d in self.destination:
            path,t=self.get_path(self.path_file[d],indexs)
            if path[0]==-9999:
                paths=None
                break
            for k in path:
                paths.append(k)
            indexs=t
        return paths

    def tensor_from_names(self, names, hfile):
        desc = [hfile[i]['global_descriptor'].__array__() for i in names]
        if self.opt.cpu:
            desc = torch.from_numpy(np.stack(desc, 0)).float()
        else:
            desc = torch.from_numpy(np.stack(desc, 0)).to(self.device).float()
        return desc

    def help(self):
        self.info = tk.Toplevel(self.master)
        self.info.geometry('800x650')
        self.info.title('Instruction')
        self.info.geometry("+{}+{}".format(self.positionRight - 300, self.positionDown - 200))
        # This will create a LabelFrame
        label_frame1 = LabelFrame(self.info, height=100, text='Steps')
        label_frame1.pack(expand='yes', fill='both')

        label1 = Label(label_frame1, text='1. Double click floor plan to pick a destination point.')
        label1.place(x=0, y=5)

        label2 = Label(label_frame1, text='2. Choose a query in query list to see localization.')
        label2.place(x=0, y=35)

        label3 = Label(label_frame1,
                       text='3. Click and see retrieved images and double click pairs to check local matches.')
        label3.place(x=0, y=65)

        label4 = Label(label_frame1, text='4. Repeat above to see other query results.')
        label4.place(x=0, y=95)

        label_frame1 = LabelFrame(self.info, height=400, text='Buttons')
        label_frame1.pack(expand='yes', fill='both', side='bottom')

        label_1 = LabelFrame(label_frame1, height=60, text='Retrieval animation')
        label_1.place(x=5, y=23)
        label1 = Label(label_1, text='Turn on or turn off the display of retrieval images')
        label1.pack()

        label_2 = LabelFrame(label_frame1, height=40, text='Plotting scale')
        label_2.place(x=5, y=70)
        label2 = Label(label_2, text='How many foot of per pixel.')
        label2.pack()

        label_3 = LabelFrame(label_frame1, height=40, text='Retrieval number')
        label_3.place(x=5, y=117)
        label3 = Label(label_3,
                       text='How many retrieved images to use.')
        label3.pack()

        label_4 = LabelFrame(label_frame1, height=60, text='Save Animation')
        label_4.place(x=5, y=164)
        label4 = Label(label_4,
                       text='Save the localization results.')
        label4.pack()

    def sequential(self):
        query_pool=[i for i in self.query_pool if i!=-9999]
        if len(query_pool)<50:
            return self.db_desc
        else:
            j=[]
            for x,y in query_pool:
                _, i_ = self.kdtree.query((x[0],y[0]),k=100)
                for i in i_:
                    if i not in j:
                        j.append(i)
            X=np.array([self.pts[i] for i in j])
            clf = IsolationForest(n_estimators=50, warm_start=True)
            clf.fit(X)
            j=[j[i] for i in clf.predict(X) if i==1]
            db_desc = self.tensor_from_names([self.db_names[i] for i in j], self.hfile)
            if len(db_desc)>0:
                return db_desc
            else:
                return self.db_desc

    def gif(self):
        c,d=int(self.e4.get()),int(self.e5.get())
        plan = Image.open(os.path.join(self.opt.query_dir, os.listdir(self.opt.query_dir)[0]))
        b, _ = plan.size
        scale = c / b
        plan = Image.open(self.data['floorplan'])
        e, _ = plan.size
        gif_shrink = e / d
        gif_ims=[]
        self.query_pool=[]
        for dic in tqdm(sorted(os.listdir(self.opt.query_dir)),desc='Generate frames'):
            dic_path = os.path.join(self.opt.query_dir, dic)
            image = Image.open(dic_path)
            query_image=image.copy()
            self.query_desc = self.extractor.feature(image)[0]
            if self.opt.cpu:
                self.query_desc = torch.from_numpy(self.query_desc).unsqueeze(0).float()
            else:
                self.query_desc = torch.from_numpy(self.query_desc).unsqueeze(0).to(self.device).float()
            sim = torch.einsum('id,jd->ij', self.query_desc, self.sequential())
            topk = torch.topk(sim, int(self.e1.get()), dim=1).indices.cpu().numpy()
            retreval_names = []
            scores = []
            sim = sim.squeeze()
            feats0 = self.extract_local_features(image)
            self.pairs = {}
            kp, lm = [], []
            for i in topk[0]:
                name = self.db_names[i].split('/')[0]
                pt0, pt1, lms ,valid= self.geometric_verification(i, feats0)
                if valid > 30:
                    for j in range(len(lms)):
                        kp.append(pt0[j])
                        lm.append(lms[j])
                    if name not in retreval_names:
                        self.pairs.update(
                            {i: {'db_image': os.path.join(self.opt.db_dir, self.db_names[i]), 'db_kpts': pt1,
                                 'q_image': os.path.join(self.opt.query_dir, dic), 'q_kpts': pt0,
                                 'score': sim[i]}})
                        retreval_names.append(self.db_names[i])
                        scores.append(sim[i])
                    else:
                        index = retreval_names.index(self.db_names[i])
                        if scores[index] < sim[i]:
                            self.pairs.update(
                                {i: {'db_image': os.path.join(self.opt.db_dir, self.db_names[i]), 'db_kpts': pt1,
                                     'q_image': os.path.join(self.opt.query_dir, dic), 'q_kpts': pt0,
                                     'score': sim[i]}})
                            scores[index] = sim[i]
            del self.query_desc, feats0
            torch.cuda.empty_cache()
            im = self.imf.copy()
            draw, im = self.prepare_image(im)
            try:
                ang_gt = self.GT[dic.split('.')[0]]['rot']
                x_gt, y_gt = self.GT[dic.split('.')[0]]['trans'][0], self.GT[dic.split('.')[0]]['trans'][1]
                x1, y1 = x_gt - 40 * np.sin(ang_gt), y_gt - 40 * np.cos(ang_gt)
                draw.ellipse((x_gt - 20, y_gt - 20, x_gt + 20, y_gt + 20), fill=(255, 0, 255))
                draw.line([(x_gt, y_gt), (x1, y1)], fill=(255, 0, 255), width=10)
            except:
                pass
            tvec, qvec = self.pnp(kp, lm)
            if len(self.query_pool)>50:
                self.query_pool.pop(0)
            if len(tvec)<=1:
                self.query_pool.append(-9999)
            if len(tvec) > 1:
                self.scale = float(self.e2.get())
                x_, _, y_ = tvec
                ang = -qvec[1] - self.rot_base
                tvec = self.T @ np.array([[x_], [y_], [1]])
                x_, y_ = tvec
                self.query_pool.append([x_,y_])
                im_ = np.array(im)
                h, _, _ = im_.shape
                im_ = cv2.putText(im_, 'Current location:  [%d,%d],  orientation:  %d degree' % (
                    x_, y_, ang*180/np.pi), (10, h - 140), cv2.FONT_HERSHEY_SIMPLEX,
                                  1, (0, 0, 255), 2, cv2.LINE_AA)
                if len(self.destination)>0:
                    for d in self.destination:
                        xx, yy = self.keyframes[d + '_00']['trans']
                        im_ = cv2.putText(im_, 'Destination location:  [%d,%d]' % (
                            xx, yy), (10, h - 200), cv2.FONT_HERSHEY_SIMPLEX,
                                          1, (0, 0, 0), 2, cv2.LINE_AA)
                im = Image.fromarray(im_)
                draw = ImageDraw.Draw(im)
                if len(self.destination)>0:
                    paths = self.cloest_path(x_, y_)
                    try:
                        if len(paths) > 1:
                            x0, y0 = self.pts[self.knames.index(paths[0])]
                            l0 = np.linalg.norm([x_ - x0, y_ - y0])
                            if (l0 < 20):
                                paths.pop(0)
                        x0, y0 = self.pts[self.knames.index(paths[0])]
                        draw.line([(x_, y_), (x0, y0)], fill=(255, 0, 255), width=5)
                        distance = np.linalg.norm([x_ - x0, y_ - y0])
                        rot = np.arctan2(x_ - x0, y_ - y0)
                        rot_ang = (rot - ang) / np.pi * 180
                        im = self.action(rot_ang, distance, im,ang,[x_,y_])
                        im = Image.fromarray(im)
                        draw = ImageDraw.Draw(im)
                        if len(paths) > 1:
                            for i in range(1, len(paths)):
                                x0, y0 = self.pts[self.knames.index(paths[i - 1])]
                                vertices = self.star_vertices([x0, y0], 15)
                                draw.polygon(vertices, fill='yellow', outline='red')
                                x1, y1 = self.pts[self.knames.index(paths[i])]
                                draw.line([(x0, y0), (x1, y1)], fill=(255, 0, 255), width=5)
                    except:
                        pass
            if self.retrieval:
                for index in retreval_names:
                    k = self.kf[index.replace('.png', '')]
                    x0, y0 = k['trans']
                    ang = k['rot']
                    x1, y1 = x0 - 20 * np.sin(ang), y0 - 20 * np.cos(ang)
                    draw.ellipse((x0 - 10, y0 - 10, x0 + 10, y0 + 10), fill=(255, 0, 0))
                    draw.line([(x0, y0), (x1, y1)], fill=(255, 0, 0), width=7)
            draw = ImageDraw.Draw(query_image)
            for p in self.matched_2D:
                xx, yy = p[0]-0.5, p[1]-0.5
                draw.rectangle([(xx - 10, yy - 10), (xx + 10, yy + 10)], fill=(0, 0, 255))
            width, height = query_image.size
            newsize = (int(width * scale), int(height * scale))
            query_image = query_image.resize(newsize)
            draw = ImageDraw.Draw(im)
            for points in self.matched_3D:
                x, y = points
                draw.rectangle([(x - 3, y - 3), (x + 3, y + 3)], fill=(0, 0, 255))

            width, height = query_image.size
            newsize = (int(width * scale), int(height * scale))
            query_image = query_image.resize(newsize)
            width, height = query_image.size
            fw, fh = im.size
            im.paste(query_image, (fw - width, fh - height))
            for i, p in enumerate(self.matched_2D):
                x2, y2 = (p[0]-0.5) * scale + fw - width, (p[1]-0.5) * scale + fh - height
                x3, y3 = self.matched_3D[i]
                draw.line([(x3, y3), (x2, y2)], fill=(0, 255, 255), width=1)
            if len(tvec) > 1:
                x1, y1 = x_ - 40 * np.sin(ang), y_ - 40 * np.cos(ang)
                draw.ellipse((x_ - 20, y_ - 20, x_ + 20, y_ + 20), fill=(50, 0, 106))
                draw.line([(x_, y_), (x1, y1)], fill=(50, 0, 106), width=10)
            fw, fh = im.size
            newsize = (int(fw / gif_shrink), int(fh / gif_shrink))
            gif_ims.append(im.resize(newsize))
        outf=str(self.e6.get())
        gif_ims[0].save(os.path.join(self.opt.topomap_path, outf+'.gif'),
                        save_all=True, append_images=gif_ims[1:], optimize=False, duration=int(self.e1.get()),
                        loop=0)
        self.win.destroy()
        showinfo("Saved!", "Animation saved to:\n{}".format(os.path.join(self.opt.topomap_path, outf+'.gif')))

    def gif_config(self):
        self.win.destroy()
        self.win = tk.Toplevel(self.master)
        self.win.wm_title("Animation setting")
        self.win.geometry('400x200')
        self.win.geometry("+{}+{}".format(self.positionRight, self.positionDown))
        self.shift = tk.Frame(self.win)
        self.shift.grid(row=0, column=0)
        self.shift1 = tk.Frame(self.shift)
        self.shift1.grid(row=0, column=0, pady=5, padx=60)
        scale = Label(self.shift1, text='Frame width in Gif:')
        scale.pack(side=LEFT)
        self.e4 = tk.Entry(self.shift1, width=4, justify='center')
        self.e4.insert(END, '450')
        self.e4.pack(side=LEFT)
        self.shift2 = tk.Frame(self.shift)
        self.shift2.grid(row=1, column=0, pady=5, padx=60)
        scale = Label(self.shift2, text='Output gif width:')
        scale.pack(side=LEFT)
        self.e5 = tk.Entry(self.shift2, width=4, justify='center')
        self.e5.insert(END, '900')
        self.e5.pack(side=LEFT)
        self.shift4 = tk.Frame(self.shift)
        self.shift4.grid(row=2, column=0, pady=5, padx=60)
        scale = Label(self.shift4, text='Frame duration:')
        scale.pack(side=LEFT)
        self.e1 = tk.Entry(self.shift4, width=2, justify='center')
        self.e1.insert(END, '30')
        self.e1.pack(side=LEFT)
        scale = Label(self.shift4, text='s')
        scale.pack(side=LEFT)

        self.shift5 = tk.Frame(self.shift)
        self.shift5.grid(row=3, column=0, pady=5, padx=60)
        scale = Label(self.shift5, text='Gif name:')
        scale.pack(side=LEFT)
        self.e6 = tk.Entry(self.shift5, width=15, justify='center')
        self.e6.insert(END, self.opt.query_dir.split('/')[-1])
        self.e6.pack(side=LEFT)
        scale = Label(self.shift5, text='.gif')
        scale.pack(side=LEFT)

        self.shift3 = tk.Frame(self.shift)
        self.shift3.grid(row=4, column=0, pady=5)
        b1 = ttk.Button(self.win, text="Go", width=6, command=self.gif)
        b1.grid(row=4, column=0, columnspan=1, padx=30, rowspan=1, sticky='w')
        b2 = ttk.Button(self.win, text="Cancel", width=6, command=self.win.destroy)
        b2.grid(row=4, column=0, columnspan=1, padx=20, rowspan=1, sticky='e')

    def gif_generator(self):
        self.win = tk.Toplevel(self.master)
        self.win.wm_title("Warning!!!")
        self.win.geometry('330x130')
        self.win.geometry("+{}+{}".format(self.positionRight, self.positionDown))
        l = tk.Label(self.win, text="Save animation to outpath?")
        l.grid(row=1, column=0, columnspan=4, rowspan=2,
               padx=20, pady=30)

        b1 = ttk.Button(self.win, text="Yes", width=6, command=self.gif_config)
        b1.grid(row=3, column=0, columnspan=1, padx=40, rowspan=3)

        b2 = ttk.Button(self.win, text="No", width=6, command=self.win.destroy)
        b2.grid(row=3, column=1, columnspan=1, padx=40, rowspan=3)

    def clear_plots(self):
        try:
            for widget in self.li.winfo_children():
                widget.destroy()
            self.li.destroy()
        except:
            pass

    def show_pairs(self):
        pairs=self.pairs[self.v.get()]
        im_q = cv2.imread(pairs['q_image'])
        im_d=cv2.imread(pairs['db_image'])
        pts_q=pairs['q_kpts']
        pts_d = pairs['db_kpts']
        h1,w1,_=im_q.shape
        h2,w2,_=im_d.shape
        scale=h2/w2
        im_d=cv2.resize(im_d,(w1,int(scale*w1)))
        h2, w2, _ = im_d.shape
        im=np.ones((h1+h2+20,w1,3),dtype=np.uint8)*255

        im[:h1,:,:]=im_q
        im[-h2:,:,:]=im_d
        width, height,_= im.shape
        for i in range(len(pts_d)):
            x0,y0=pts_q[i]
            x1,y1=pts_d[i]
            im=cv2.line(im,(int(x0),int(y0)),(int(x1),int(y1)+h1+20),color=[0,255,0],thickness=1)
        self.pairs_image = Image.fromarray(im.copy()[:,:,::-1])
        scale = 210 / width
        newsize = [210, int(height * scale)]
        im = cv2.resize(im,(newsize[1],newsize[0]))
        im = Image.fromarray(im[:,:,::-1])
        tkimage1 = ImageTk.PhotoImage(im)
        self.myvar1 = Label(self, image=tkimage1)
        self.myvar1.bind('<Double-Button-1>', self.new_window_pairs)
        self.myvar1.image = tkimage1
        self.myvar1.grid(row=18, column=0, columnspan=1, padx=10, rowspan=20, sticky="snew")
        k=self.kf[pairs['db_image'].split('/')[-1].replace('.png','')]

        im=self.fp.copy()
        draw = ImageDraw.Draw(im)
        x_, y_ = k['trans']
        phi = k['rot']

        draw.ellipse((x_ - 9, y_ - 9, x_ + 9, y_ + 9), fill=(255, 0, 0))
        rr = np.pi / 2 + phi
        x1bar = x_ - self.rbar * np.sin(phi)
        y1bar = y_ - self.rbar * np.cos(phi)
        draw.line([(x1bar, y1bar), (x_, y_)], fill=(255, 0, 0), width=7)
        x1 = x1bar - self.rhat * np.sin(rr)
        y1 = y1bar - self.rhat * np.cos(rr)
        corners = [[x1, y1]]
        for i in range(1, self.n):
            x1, y1 = corners[-1]
            rr += self.gamma
            corners.append([x1 - 2 * self.rhat * np.sin(rr), y1 - 2 * self.rhat * np.cos(rr)])
        corners = np.array(corners)
        draw = ImageDraw.Draw(im)
        for i in range(1, self.n):
            draw.line([(corners[i - 1][0], corners[i - 1][1]), (corners[i][0], corners[i][1])], fill=(135, 206, 235),
                      width=7)
        draw.line([(corners[self.n - 1][0], corners[self.n - 1][1]), (corners[0][0], corners[0][1])], fill=(255, 0, 0), width=7)
        width, height = im.size
        scale = 1600 / width
        newsize = (1600, int(height * scale))
        im = im.resize(newsize)
        tkimage1 = ImageTk.PhotoImage(im)
        self.myvar2 = Label(self, image=tkimage1)
        self.myvar2.image = tkimage1
        self.myvar2.grid(row=0, column=4, columnspan=1, rowspan=40, sticky="snew")

    def new_window_pairs(self,w):
        self.newWindow = tk.Toplevel(self)
        self.app1 = Pairs_window(self.newWindow,self.pairs_image)

    def plot_pairs(self):
        self.clear_plots()
        self.li = ttk.LabelFrame(self, text="%d Database frames" % len(self.db_names))
        self.li1 = ttk.LabelFrame(self.li, text='Frame name')
        self.li2 = ttk.LabelFrame(self.li, text='Score/Valid')
        self.li1.pack(side=LEFT)
        self.li2.pack(side=RIGHT)
        self.li.grid(row=19, column=0, padx=10, rowspan=1, sticky="ew")
        self.v = tk.IntVar()
        for i in self.pairs.keys():
            name=self.pairs[i]['db_image'].split('/')[-1]
            self.lb2=tk.Radiobutton(self.li1,
                           text='%s' % name,
                           padx=20,
                           command=self.show_pairs,
                           variable=self.v,
                           value=i).pack(anchor=tk.W)
            self.sl2 = Label(self.li2, text='%1.2f/%03d' % (self.pairs[i]['score'],self.pairs[i]['valid_num']))
            self.sl2.pack()

    def extract_local_features(self, image0):
        data0 = self.prepare_data(image0)
        pred0 = self.sp_model(data0.to(self.device))
        del data0
        torch.cuda.empty_cache()
        pred0 = {k: v[0].cpu().detach().numpy() for k, v in pred0.items()}
        if 'keypoints' in pred0:
            pred0['keypoints'] = (pred0['keypoints'] + .5) - .5
        pred0.update({'image_size': np.array([image0.size[0], image0.size[1]])})
        return pred0

    def colmap2world(self,tvec, quat):
        r = R.from_quat(quat)
        rmat = r.as_matrix()
        rmat = rmat.transpose()
        rot=R.from_matrix(r.as_matrix().transpose()).as_rotvec()
        return  -np.matmul(rmat, tvec).reshape(3), rot

    def pnp(self,kpts,lms):
        kpts=[np.array([[i[0]+0.5],[i[1]+0.5]],dtype=float) for i in kpts]
        lms=[np.array([[self.landmarks[str(i)]['x']],[self.landmarks[str(i)]['y']],[self.landmarks[str(i)]['z']]],dtype=float) for i in lms]
        ret = pycolmap.absolute_pose_estimation(kpts, lms, self.camera_model, self.ransac_threshold)
        self.matched_2D,self.matched_3D=[],[]
        if ret['success']:
            matched_3D = np.array(lms)[ret['inliers']].squeeze()
            self.matched_3D=np.ones((3,matched_3D.shape[0]))
            self.matched_3D[0,:]=matched_3D[:,0]
            self.matched_3D[1, :] = matched_3D[:, 2]
            self.matched_3D=(self.T@self.matched_3D).T
            self.matched_2D = np.array(kpts)[ret['inliers']].squeeze()
            qvec=ret['qvec']
            qvec=[qvec[1],qvec[2],qvec[3],qvec[0]]
            tvec=ret['tvec']
            tvec,qvec=self.colmap2world(tvec,qvec)
            return tvec,qvec
        return [-1],[-1]

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

    def geometric_verification(self, i, feats0):
        # tm = time.time()
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
        pts0, pts1 ,lms= [], [],[]
        index_list=self.kf[self.db_names[i].replace('.png', '')]['kp_index']
        for n, m in enumerate(matches):
            if (m != -1)and(m in index_list):
                pts0.append(feats0['keypoints'][n].tolist())
                pts1.append(feats1['keypoints'][m].tolist())
                lms.append(self.kf[self.db_names[i].replace('.png', '')]['lm_ids'][index_list.index(m)])
        del data, feats1, pred
        # tm2=time.time()
        # tm3 = tm2-tm

        try:
            pts0_ = np.int32(pts0)
            pts1_ = np.int32(pts1)
            F, mask = cv2.findFundamentalMat(pts0_, pts1_, cv2.RANSAC)
            _, inliers = ransac(
                (pts0_, pts1_),
                AffineTransform,
                min_samples=3,
                residual_threshold=20,
                max_trials=1000)

            # valid = sum(inliers)
            valid = len(pts0_[mask.ravel() == 1])
        except:
            valid=0
        # print(tm3,time.time()-tm2)
        torch.cuda.empty_cache()
        pt0, pt1=pts0, pts1
        return [pt0, pt1,lms,valid]

    def action(self,rot_ang,distance,image):
        im=np.array(image)
        h,_,_=im.shape
        rot_ang=(-rot_ang)%360
        rot_clock = round(rot_ang / 30) % 12
        im = cv2.putText(im, u'Please walk %.1f meters along %d clock\n' % (distance * self.scale * 0.3048, rot_clock), (10, h - 80), cv2.FONT_HERSHEY_SIMPLEX,
                         1.2, (255, 0, 0), 2, cv2.LINE_AA)
        # if abs(rot_ang)>360:
        #     rot_ang=np.sign(rot_ang)*(abs(rot_ang)-abs(rot_ang//360)*360)
        # if rot_ang >= 0:
        #     if rot_ang>180:
        #         im = cv2.putText(im, u'Please right turn %3d degree and walk %4dcm' % (
        #             360-rot_ang, distance * self.scale * 30.48), (10, h-80), cv2.FONT_HERSHEY_SIMPLEX,
        #                          1.2, (255, 0, 0), 2, cv2.LINE_AA)
        #     else:
        #         im = cv2.putText(im, u'Please left turn %3d degree and walk %4dcm' % (
        #             rot_ang, distance * self.scale * 30.48),(10, h-80), cv2.FONT_HERSHEY_SIMPLEX,
        #                          1.2, (255, 0, 0), 2, cv2.LINE_AA)
        # else:
        #     if rot_ang<-180:
        #         im = cv2.putText(im, u'Please left turn %3d degree and walk %4dcm' % (
        #             360+rot_ang, distance * self.scale * 30.48),(10, h-80), cv2.FONT_HERSHEY_SIMPLEX,
        #                          1.2, (255, 0, 0), 2, cv2.LINE_AA)
        #     else:
        #         im = cv2.putText(im, u'Please right turn %3d degree and walk %4dcm' % (
        #             -rot_ang, distance * self.scale * 30.48), (10, h - 80), cv2.FONT_HERSHEY_SIMPLEX,
        #                          1.2, (255, 0, 0), 2, cv2.LINE_AA)
        return im

    def prepare_image(self,image):
        draw=ImageDraw.Draw(image)
        l=60*self.plot_scale
        x_,y_=50*self.plot_scale,l
        ang=0
        x1, y1 = x_ - 40*self.plot_scale * np.sin(ang), y_ - 40*self.plot_scale * np.cos(ang)
        draw.ellipse((x_ - 20*self.plot_scale, y_ - 20*self.plot_scale, x_ + 20*self.plot_scale, y_ + 20*self.plot_scale), fill=(50, 0, 106))
        draw.line([(x_, y_), (x1, y1)], fill=(50, 0, 106), width=int(10*self.plot_scale))
        im=np.array(image)
        im=cv2.putText(im, 'Estimation pose', (int(100*self.plot_scale), int(l)), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 0), round(2*self.plot_scale), cv2.LINE_AA)
        image=Image.fromarray(im)
        draw = ImageDraw.Draw(image)
        if self.GT:
            l+=70*self.plot_scale
            x_, y_ = 50*self.plot_scale, l
            x1, y1 = x_ - 40*self.plot_scale * np.sin(ang), y_ - 40*self.plot_scale * np.cos(ang)
            draw.ellipse((x_ - 20*self.plot_scale, y_ - 20*self.plot_scale, x_ + 20*self.plot_scale, y_ + 20*self.plot_scale), fill=(255, 0, 255))
            draw.line([(x_, y_), (x1, y1)], fill=(255, 0, 255), width=int(10*self.plot_scale))
            im = np.array(image)
            im = cv2.putText(im, 'Ground truth pose', (int(100*self.plot_scale), int(l)), cv2.FONT_HERSHEY_SIMPLEX,
                             1, (0, 0, 0), round(2**self.plot_scale), cv2.LINE_AA)
            image = Image.fromarray(im)
            draw = ImageDraw.Draw(image)
        if self.retrieval:
            l+=70*self.plot_scale
            x_, y_ = 50*self.plot_scale, l
            x1, y1 = x_ - 20*self.plot_scale * np.sin(ang), y_ - 20 *self.plot_scale* np.cos(ang)
            draw.ellipse((x_ - 10*self.plot_scale, y_ - 10*self.plot_scale, x_ + 10*self.plot_scale, y_ + 10*self.plot_scale), fill=(255, 0, 0))
            draw.line([(x_, y_), (x1, y1)], fill=(255, 0, 0), width=int(7*self.plot_scale))
            im = np.array(image)
            im = cv2.putText(im, 'Similar images', (int(100*self.plot_scale), int(l)), cv2.FONT_HERSHEY_SIMPLEX,
                             1, (0, 0, 0), round(2*self.plot_scale), cv2.LINE_AA)
            image = Image.fromarray(im)
            draw = ImageDraw.Draw(image)
        if len(self.destination)>0:
            l+=70*self.plot_scale
            vertices = self.star_vertices([50*self.plot_scale, l], 30)
            draw.polygon(vertices, fill='red')
            im = np.array(image)
            im = cv2.putText(im, 'Destination', (int(100*self.plot_scale), int(l)), cv2.FONT_HERSHEY_SIMPLEX,
                             1, (0, 0, 0), round(2*self.plot_scale), cv2.LINE_AA)
            image = Image.fromarray(im)
            draw = ImageDraw.Draw(image)
        draw.rectangle([(10,5),(400+100*self.plot_scale,l+40*self.plot_scale)],outline='black',width=int(2*self.plot_scale))
        return draw,image

    def FloorPlan_select(self, action):
        # tm=time.time()
        self.value = self.lb.get(self.lb.curselection())
        if action == 'up':
            i = self.dics.index(self.value)
            if i > 0:
                self.value = self.dics[i - 1]
        if action == 'down':
            i = self.dics.index(self.value)
            if i < (len(self.dics) - 1):
                self.value = self.dics[i + 1]
        for type in self.globs:
            q_path=os.path.join(self.opt.query_dir, self.value + type)
            if os.path.exists(q_path):
                image = Image.open(q_path)
                break
        width, height = image.size
        scale = 210 / width
        newsize = (210, int(height * scale))
        im=image.copy()
        im = im.resize(newsize)
        tkimage1 = ImageTk.PhotoImage(im)
        self.myvar1 = Label(self, image=tkimage1)
        self.myvar1.image = tkimage1
        self.myvar1.grid(row=1, column=0, columnspan=1, padx=10, rowspan=5, sticky="snew")
        scale = 640 / width
        newsize = (640, int(height * scale))
        image = image.resize(newsize)
        self.query_desc = self.extractor.feature(image)[0]
        if self.opt.cpu:
            self.query_desc = torch.from_numpy(self.query_desc).unsqueeze(0).float()
        else:
            self.query_desc = torch.from_numpy(self.query_desc).unsqueeze(0).to(self.device).float()
        sim = torch.einsum('id,jd->ij', self.query_desc, self.db_desc)
        topk = torch.topk(sim, int(self.e1.get()), dim=1).indices.cpu().numpy()
        retreval_names = []
        scores = []
        sim = sim.squeeze()
        feats0 = self.extract_local_features(image)
        self.pairs = {}
        kp,lm=[],[]
        # retreval_pts = []
        score=[]
        total_score=0
        max_valid=0
        max_pts=None
        # tt1,tt2=0,0
        # print('image retrieval time:',time.time()-tm)
        for i in topk[0]:
            name = self.db_names[i].split('/')[0]
            pt0, pt1,lms,valid= self.geometric_verification(i, feats0)
            # tt1+=ttt1
            # tt2+=ttt2
            if valid > 30:
                # if max_valid<valid:
                #     max_valid=valid
                #     max_pts=self.keyframes[name.replace('.png', '')]['trans']
                # if valid > 75:
                #     # retreval_pts.append(self.keyframes[name.replace('.png', '')]['trans'])
                #     total_score+=valid
                #     score.append(valid)
                for j in range(len(lms)):
                    kp.append(pt0[j])
                    lm.append(lms[j])
                if name not in retreval_names:
                    self.pairs.update({i:{'db_image': os.path.join(self.opt.db_dir, self.db_names[i]), 'db_kpts': pt1,
                                       'q_image': q_path,'q_kpts':pt0,'valid_num':valid,'score':sim[i]}})
                    retreval_names.append(self.db_names[i])
                    scores.append(sim[i])
                else:
                    index = retreval_names.index(self.db_names[i])
                    if scores[index] < sim[i]:
                        self.pairs.update(
                            {i: {'db_image': os.path.join(self.opt.db_dir, self.db_names[i]), 'db_kpts': pt1,
                                 'q_image': q_path, 'q_kpts': pt0,'valid_num':valid,'score':sim[i]}})
                        scores[index] = sim[i]
        # print('local descriptor match time:',round(tt2,5),'geometric verification time:',round(tt1,5))
        # if len(retreval_pts)>0:
        #     for i,x in enumerate(retreval_pts):
        #         retreval_pts[i]=np.array(x)*score[i]/total_score
        # else:
        #     retreval_pts.append(max_pts)
        # xx_,yy_=np.sum(np.array(retreval_pts),0)
        del self.query_desc, feats0
        torch.cuda.empty_cache()
        im=self.imf.copy()
        # self.temp_image=im.copy()
        draw,im= self.prepare_image(im)
        try:
            ang_gt = self.GT[self.value]['rot']
            x_gt, y_gt = self.GT[self.value]['trans'][0], self.GT[self.value]['trans'][1]
            x1, y1 = x_gt - 40 * np.sin(ang_gt), y_gt - 40 * np.cos(ang_gt)
            draw.ellipse((x_gt - 20*self.plot_scale, y_gt - 20*self.plot_scale, x_gt + 20*self.plot_scale, y_gt + 20*self.plot_scale), fill=(255, 0, 255))
            draw.line([(x_gt, y_gt), (x1, y1)], fill=(255, 0, 255), width=int(10*self.plot_scale))
        except:
            pass
        # single image
        # tvec, qvec = self.pose_refine(out, p2d_inlier, p3d_inlier)
        #multiple images
        if len(kp)>0:
            out, p2d_inlier, p3d_inlier=self.coarse_pose(kp,lm,np.array([width / 2, height / 2]))
            self.list_2d.append(p2d_inlier)
            self.list_3d.append(p3d_inlier)
            self.initial_poses.append(out['pose'])
            self.pps.append(out['pp'])
            tvecs, qvecs = self.pose_multi_refine(self.list_2d, self.list_3d, self.initial_poses, self.pps)
            tvec, qvec = tvecs[-1], qvecs[-1]
            print(tvec)
        # print('total time:', round(time.time() - tm,5))
        # if len(tvec)>1:
            self.scale = float(self.e2.get())
            x_,_,y_=tvec
            ang = -qvec[1] - self.rot_base
            tvec=self.T@np.array([[x_],[y_],[1]])
            x_,y_=tvec
            # x_,y_=xx_,yy_
            im_ = np.array(im)
            h, _, _ = im_.shape
            an=ang*180/np.pi
            if an < 0.5:
                an += 360
            an=360-an
            im_ = cv2.putText(im_, 'Current location:  [%d,%d],  orientation:  %d degree' % (
                x_, y_, an), (10, h - 200), cv2.FONT_HERSHEY_SIMPLEX,
                             1, (0, 0, 255), 2, cv2.LINE_AA)
            if len(self.destination)>0:
                for ind,d in enumerate(self.destination):
                    xx,yy=self.keyframes[d+'_00']['trans']
                    im_ = cv2.putText(im_, 'Destination location %d:  [%d,%d]' % (ind+1,
                        xx,yy), (10, h - 140-(len(self.destination)-ind-1)*60), cv2.FONT_HERSHEY_SIMPLEX,
                                      1, (0, 0, 0), 2, cv2.LINE_AA)
            # if self.temp_image:
            #     im=self.temp_image
            # else:
            im= Image.fromarray(im_)
            draw = ImageDraw.Draw(im)
            try:
                error_rot=(ang_gt-ang)/np.pi*180
                error_trans=((x_-x_gt)**2+(y_-y_gt)**2)**(0.5)
                self.lii = Label(self, text=u'Error: t: %4.2fcm, r: %3.2f\N{DEGREE SIGN}' % (
                error_trans * self.scale * 30.48, error_rot))
                self.lii.grid(row=13, column=0, padx=10, pady=1, sticky='we')
            except:
                pass
            if len(self.destination)>0:
                paths=self.cloest_path(x_,y_)
                try:
                    if len(paths)>1:
                        x0,y0=self.pts[self.knames.index(paths[0])]
                        # x1,y1=self.pts[self.knames.index(paths[1])]
                        l0=np.linalg.norm([x_-x0,y_-y0])
                        # l1=np.linalg.norm([x0-x1,y0-y1])
                        if (l0<20):
                            paths.pop(0)
                    color = self.colors[0]
                    self.colors = self.colors[1:]
                    x0, y0 = self.pts[self.knames.index(paths[0])]
                    draw.line([(x_, y_), (x0, y0)], fill=(255,0,0), width=int(10*self.plot_scale))
                    distance=np.linalg.norm([x_-x0,y_-y0])
                    rot=np.arctan2(x_-x0,y_-y0)
                    rot_ang=(rot-ang)/np.pi*180
                    im=self.action(rot_ang[0],distance,im)
                    im = Image.fromarray(im)
                    draw = ImageDraw.Draw(im)
                    if len(paths) > 1:
                        for i in range(1, len(paths)):
                            x0, y0 = self.pts[self.knames.index(paths[i - 1])]
                            vertices = self.star_vertices([x0, y0], 15)
                            draw.polygon(vertices, fill='yellow',outline='red')
                            x1, y1 =self.pts[self.knames.index(paths[i])]
                            draw.line([(x0, y0), (x1, y1)], fill=(255,0,0), width=int(10*self.plot_scale))
                except:
                    pass
            x1, y1 = x_ - 40*self.plot_scale * np.sin(ang), y_ - 40*self.plot_scale * np.cos(ang)
            draw.ellipse((x_ - 20*self.plot_scale, y_ - 20*self.plot_scale, x_ + 20*self.plot_scale, y_ + 20*self.plot_scale), fill=(50, 0, 106))
            draw.line([(x_, y_), (x1, y1)], fill=(50, 0, 106), width=int(10*self.plot_scale))
            # self.temp_image = im.copy()

        self.fp = im.copy()
        if self.retrieval:
            for index in retreval_names:
                k = self.kf[index.replace('.png','')]
                x_, y_ = k['trans']
                ang=k['rot']
                x1, y1 = x_ - 20*self.plot_scale * np.sin(ang), y_ - 20*self.plot_scale * np.cos(ang)
                draw.ellipse((x_ - 10*self.plot_scale, y_ - 10*self.plot_scale, x_ + 10*self.plot_scale, y_ + 10*self.plot_scale), fill=(255, 0, 0))
                draw.line([(x_, y_), (x1, y1)], fill=(255, 0, 0), width=int(7*self.plot_scale))
        width, height = im.size
        scale = 1600 / width
        newsize = (1600, int(height * scale))
        im = im.resize(newsize)
        tkimage1 = ImageTk.PhotoImage(im)
        self.myvar1 = Label(self, image=tkimage1)
        self.myvar1.image = tkimage1
        self.myvar1.grid(row=0, column=4, columnspan=1, rowspan=40, sticky="snew")
        self.plot_pairs()
        self.myvar1.bind('<Double-Button-1>', lambda event, action='double':
        self.show_trajectory(action))
        query_names=os.listdir(opt.query_dir)
        for q in query_names:
            if q not in self.query_names:
                self.query_names=query_names
                self.set_query(self.query_names)
                break
        for q in self.query_names:
            if q not in query_names:
                self.set_query(query_names)
                break

    def show_trajectory(self,w):
        self.newWindow = tk.Toplevel(self.master)
        self.app1 = Trajectory_window(self.newWindow, parent=self)

def main(opt):
    data = load_data(opt.topomap_path)
    if os.path.exists(opt.GT_path):
        GT = load_data(opt.GT_path)
        GT_trajectory = {}
        for k, v in GT['keyframes'].items():
            GT_trajectory.update({k: {'trans': GT['keyframes'][k]['trans'], 'rot': GT['keyframes'][k]['rot']}})
    else:
        GT_trajectory=None
        print('No query ground truth')
    style = Style(theme='darkly')
    root = style.master
    Main_window(root, data, GT_trajectory, opt)
    root.mainloop()


if __name__ == '__main__':
    opt = options()
    main(opt)