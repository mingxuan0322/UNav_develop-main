import math
import warnings
import collections
import tkinter as tk
from tkinter import *
import numpy as np
from tkinter import ttk,filedialog
from ttkbootstrap import Style
from tkinter.messagebox import showinfo
from PIL import Image,ImageDraw, ImageTk
from shapely.geometry import LineString
import argparse
import os
import cv2
import json
from tqdm import tqdm

def options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work_path', default=None, required=True,
                        help='work path')
    parser.add_argument('--Place', default=None, required=True,
                        help='which Place')
    parser.add_argument('--Building', default=None, required=True,
                        help='which building')
    parser.add_argument('--Floor', default=None, required=True,
                        help='which floor')
    parser.add_argument('--plan', default=None, required=True,
                        help='Floor plan direction')
    opt = parser.parse_args()
    return opt

def load_data(path):
    if os.path.exists(os.path.join(path, 'slam_data.json')):
        with open(os.path.join(path, 'slam_data.json'), 'r') as f:
            data=json.load(f)
            keyframes = data['keyframes']
            pts = np.array(
                [keyframes[k]['trans'] for k in list(keyframes.keys())], dtype=int)
            knames = [k.split('_')[0] for k in list(keyframes.keys())]
            # plan = data['floorplan']
    else:
        print('Map doesn\'t exist!')
        exit()
    return knames,pts

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

class Draw_window:
    """ Display and zoom image """
    def __init__(self, placeholder,parent,w,h,removing=True):
        """ Initialize the ImageFrame """
        self.removing=removing
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
        self.coords = {"x":0,"y":0,"x2":0,"y2":0}
        hbar.configure(command=self.__scroll_x)  # bind scrollbars to the canvas
        vbar.configure(command=self.__scroll_y)
        # Bind events to the Canvas
        self.canvas.bind('<Configure>', lambda event: self.__show_image())  # canvas is resized
        self.canvas.bind('<Button-1>', self.__move_from)  # remember canvas position
        self.canvas.bind('<B1-Motion>',     self.__move_to)  # move canvas to the new position
        self.canvas.bind('<MouseWheel>', self.__wheel)  # zoom for Windows and MacOS, but not Linux
        self.canvas.bind('<Button-5>',   self.__wheel)  # zoom for Linux, wheel scroll down
        self.canvas.bind('<Button-4>',   self.__wheel)  # zoom for Linux, wheel scroll up
        self.canvas.bind('<ButtonPress-3>', self.__click)
        self.canvas.bind('<B3-Motion>', self.__drag)
        self.canvas.bind('<ButtonRelease-3>', self.__release)
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
            self.__image = Image.new('RGB',(h,w))  # open image, but down't load it
        self.imwidth, self.imheight = w,h  # public for outer classes
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
        self.__pyramid = [self.smaller()] if self.__huge else [self.__image]
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
        self.r=5
        self.chose={}
        self.nochose={}
        self.p_chose={}
        self.p_nochose={}
        self.add_chose={}
        self.add_rectangle={}
        self.parent.exist_lines=[]
        self.l=[]
        for l in tqdm(self.parent.lines.tolist()):
            x0,y0,x1,y1=l
            if l in self.parent.removed_lines:
                self.chose.update({str(x0) + str(y0) + str(x1) + str(y1): {
                    'index': self.canvas.create_line(x0, y0, x1, y1, width=self.r, fill='purple', activefill='yellow'),
                    'line': l}})
            else:
                self.parent.exist_lines.append(l)
                self.nochose.update({str(x0) + str(y0) + str(x1) + str(y1): {
                    'index': self.canvas.create_line(x0, y0, x1, y1, width=2 * self.r, fill='yellow',
                                                     activefill='purple'), 'line': l}})
        if not self.removing:
            for l in self.parent.add_line:
                x0, y0, x1, y1 = l
                self.add_chose.update({str(x0) + str(y0) + str(x1) + str(y1): {
                    'index': self.canvas.create_line(x0, y0, x1, y1, width=2 * self.r, fill='red', activefill='green'),
                    'line': l}})
        else:
            for l in self.parent.clear_rectangle.keys():
                x0, y0, x1, y1 =self.parent.clear_rectangle[l]['coordinates']
                lines=self.parent.clear_rectangle[l]['lines']
                self.add_rectangle.update({str(x0) + str(y0) + str(x1) + str(y1): {
                    'index': [self.canvas.create_line(x0, y0, x1, y0,
                                        width=self.r * 2, fill='red', activefill='green'),
                              self.canvas.create_line(x0, y0, x0, y1,
                                                      width=self.r * 2, fill='red', activefill='green'),
                              self.canvas.create_line(x1, y0, x1, y1,
                                                      width=self.r * 2, fill='red', activefill='green'),
                              self.canvas.create_line(x0, y1, x1, y1,
                                                      width=self.r * 2, fill='red', activefill='green'),
                              ],
                    'line': lines}})
        for i in range(len(self.parent.knames)):
            x0, y0 = self.parent.pts[i]
            if (str(x0) + '-' + str(y0)) not in self.parent.destination:
                self.p_nochose.update({str(x0) + '-' + str(y0): {
                    'index': self.canvas.create_oval(x0 - self.r, y0 - self.r, x0 + self.r, y0 + self.r, fill='green',
                                                     activefill='red'), 'id': self.parent.knames[i],'text_id':''}})
            else:
                self.p_chose.update({str(x0) + '-' + str(y0): {
                    'index': self.canvas.create_oval(x0 - 2*self.r, y0 - 2*self.r, x0 + 2*self.r, y0 + 2*self.r, fill='red',
                                                     activefill='green'), 'id': self.parent.destination[str(x0) + '-' + str(y0)]['id'],'text_id':self.canvas.create_text(x0, y0 + self.r * 3,
                                                            fill="red", font="Times 20 italic bold",
                                                            text=self.parent.destination[str(x0) + '-' + str(y0)]['name'])}})
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

    def judge_intersection(self,a, b, lines):
        x0,y0,x1,y1=lines
        minx,maxx=sorted([a[0],b[0]])
        miny,maxy=sorted([a[1],b[1]])
        if (x0>minx)and(x0<maxx)and(y0>miny)and(y0<maxy):
            return True
        if (x1>minx)and(x1<maxx)and(y1>miny)and(y1<maxy):
            return True
        other = LineString([(x0,y0), (x1,y1)])
        line = LineString([(a[0], a[1]), (a[0], b[1])])
        if line.intersects(other):
            return True
        line = LineString([(a[0], a[1]), (b[0], a[1])])
        if line.intersects(other):
            return True
        line = LineString([(b[0], a[1]), (b[0], b[1])])
        if line.intersects(other):
            return True
        line = LineString([(a[0],b[1]), (b[0], b[1])])
        if line.intersects(other):
            return True
        return False

    def __release(self,event):
        bbox = self.canvas.coords(self.container)
        scale = (bbox[2] - bbox[0]) / self.imwidth
        x00, y00, x10, y10 = (self.coords["x"] - bbox[0]) / scale, (self.coords["y"] - bbox[1]) / scale, (
                    self.coords["x2"] - bbox[0]) / scale, (self.coords["y2"] - bbox[1]) / scale
        if self.removing:
            l=[]
            for x0,y0,x1,y1 in self.parent.lines:
                if self.judge_intersection([x00,y00],[x10,y10],[x0,y0,x1,y1]):
                    x0_, x1_, y0_, y1_ = x0 * scale + bbox[0], x1 * scale + bbox[0], y0 * scale + bbox[1], y1 * scale + \
                                         bbox[1]
                    if str(x0) + str(y0) + str(x1) + str(y1) in self.nochose:
                        self.chose.update({str(x0) + str(y0) + str(x1) + str(y1): {
                            'index': self.canvas.create_line(x0_, y0_, x1_, y1_, width=self.r, fill='purple',
                                                             activefill='yellow'),
                            'line': self.nochose[str(x0) + str(y0) + str(x1) + str(y1)]['line']}})
                        self.canvas.delete(self.nochose[str(x0) + str(y0) + str(x1) + str(y1)]['index'])
                        self.nochose.pop(str(x0) + str(y0) + str(x1) + str(y1))
                        self.parent.exist_lines.remove([x0, y0, x1, y1])
                        self.parent.removed_lines.append([x0,y0,x1,y1])
                    l.append([x0, y0, x1, y1])
            self.add_rectangle.update(
                {str(x00) + str(y00) + str(x10) + str(y10): {
                    'index': self.l[-1],
                    'lines':l}})
            self.parent.clear_rectangle.update({str(x00) + str(y00) + str(x10) + str(y10): {
                    'coordinates': [x00, y00, x10, y10],
                    'lines':l}})
            self.parent.plot()
        else:
            try:
                self.add_chose.update(
                    {str(x00) + str(y00) + str(x10) + str(y10): {
                        'index': self.l[-1],
                        'line': [x00,y00,x10,y10]}})
                self.parent.add_line.append([x00,y00,x10,y10])
                self.parent.plot()
            except:
                pass

    def __drag(self,event):
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        self.coords["x2"] = x
        self.coords["y2"] = y
        if self.removing:
            self.canvas.coords(self.l[-1][0], self.coords["x"], self.coords["y"], self.coords["x"], self.coords["y2"])
            self.canvas.coords(self.l[-1][1], self.coords["x"], self.coords["y"], self.coords["x2"], self.coords["y"])
            self.canvas.coords(self.l[-1][2], self.coords["x2"], self.coords["y"], self.coords["x2"], self.coords["y2"])
            self.canvas.coords(self.l[-1][3], self.coords["x"], self.coords["y2"], self.coords["x2"], self.coords["y2"])
        else:
            self.canvas.coords(self.l[-1], self.coords["x"],self.coords["y"],self.coords["x2"],self.coords["y2"])

    def __click(self,event):
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        self.coords["x"] = x
        self.coords["y"] = y
        bbox = self.canvas.coords(self.container)
        scale = (bbox[2] - bbox[0]) / self.imwidth
        x_, y_ = (x - bbox[0]) / scale, (y - bbox[1]) / scale

        if self.removing:
            self.l.append(
                [self.canvas.create_line(self.coords["x"], self.coords["y"], self.coords["x"], self.coords["y"],
                                        width=self.r * 2, fill='red', activefill='green') for i in range(4)])
        else:
            for s in self.p_chose.keys():
                list1 = s.split('-')
                x0, y0 = float(list1[0]), float(list1[1])
                if (np.linalg.norm((x_ - x0, y_ - y0)) < self.r * 2):
                    if s not in self.parent.destination:
                        self.parent.destination.update({s: {'id': self.p_chose[s]['id']}})
                        self.parent.name_destination(s)
                        if self.text:
                            self.canvas.delete(self.text)
                    return 0
            for s in self.p_nochose.keys():
                list1 = s.split('-')
                x0, y0 = float(list1[0]), float(list1[1])
                if (np.linalg.norm((x_ - x0, y_ - y0)) < self.r):
                    if s in self.parent.destination:
                        self.parent.remove_destination(s)
                        ID=self.p_nochose[s]['id']
                        self.canvas.delete(self.p_nochose[s]['index'])
                        self.canvas.delete(self.p_nochose[s]['text_id'])
                        x0l, x0r, y0l, y0r = (x0 - self.r) * scale + bbox[0], (x0 + self.r) * scale + bbox[0], (
                                y0 - self.r) * scale + bbox[1], (y0 + self.r) * scale + bbox[1]
                        self.p_nochose.update({s: {
                            'index': self.canvas.create_oval(x0l, y0l, x0r, y0r, fill='green', activefill='red'),
                            'id': ID}})
                    return 0
            self.l.append(self.canvas.create_line(self.coords["x"], self.coords["y"], self.coords["x"], self.coords["y"],width=self.r*2, fill='red', activefill='green'))

    def __rotate_coordinates(self,x, y, ox, oy, a):
        return (x - ox) * np.cos(a) - (y - oy) * np.sin(a) + ox, (x - ox) * np.sin(a) + (y - oy) * np.cos(a) + oy

    def __move_from(self, event):
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        bbox = self.canvas.coords(self.container)
        scale = (bbox[2] - bbox[0]) / self.imwidth
        x_, y_ = (x - bbox[0]) / scale, (y - bbox[1]) / scale

        if not self.removing:
            for i in range(len(self.parent.knames)):
                x0, y0 = self.parent.pts[i]
                if (np.linalg.norm((x_ - x0, y_ - y0)) < self.r * 2) and (str(x0) + '-' + str(y0) in self.p_chose):
                    x0l, x0r, y0l, y0r = (x0 - self.r) * scale + bbox[0], (x0 + self.r) * scale + bbox[0], (
                            y0 - self.r) * scale + bbox[1], (y0 + self.r) * scale + bbox[1]
                    if str(x0) + '-' + str(y0) in self.parent.destination:
                        self.p_nochose.update({str(x0) + '-' + str(y0): {
                            'index': self.canvas.create_oval(x0l, y0l, x0r, y0r, fill='red', activefill='red'),
                            'text_id':self.p_chose[str(x0) + '-' + str(y0)]['text_id'],
                            'id': self.parent.knames[i]}})
                    else:
                        self.p_nochose.update({str(x0) + '-' + str(y0): {
                            'index': self.canvas.create_oval(x0l, y0l, x0r, y0r, fill='green', activefill='red'),
                            'text_id': self.p_chose[str(x0) + '-' + str(y0)]['text_id'],
                            'id': self.parent.knames[i]}})
                    self.canvas.delete(self.p_chose[str(x0) + '-' + str(y0)]['index'])
                    self.p_chose.pop(str(x0) + '-' + str(y0))
                    break
                if (np.linalg.norm((x_ - x0, y_ - y0)) < self.r) and (str(x0) + '-' + str(y0) in self.p_nochose):
                    if self.text:
                        self.canvas.delete(self.text)
                    x0l, x0r, y0l, y0r = (x0 - self.r * 2) * scale + bbox[0], (x0 + self.r * 2) * scale + bbox[0], (
                            y0 - self.r * 2) * scale + bbox[1], (y0 + self.r * 2) * scale + bbox[1]
                    self.text = self.canvas.create_text(x0 * scale + bbox[0], (y0 + self.r * 3) * scale + bbox[1],
                                                        fill="darkblue", font="Times 10 italic bold",
                                                        text="Right click to pick")
                    self.p_chose.update({str(x0) + '-' + str(y0): {
                        'index': self.canvas.create_oval(x0l, y0l, x0r, y0r, fill='red', activefill='green'),
                        'text_id': self.p_nochose[str(x0) + '-' + str(y0)]['text_id'],
                        'id': self.parent.knames[i]}})
                    self.canvas.delete(self.p_nochose[str(x0) + '-' + str(y0)]['index'])
                    self.p_nochose.pop(str(x0) + '-' + str(y0))
                    x_, y_ = x0, y0
                    if len(self.p_chose) > 1:
                        for i in list(self.p_chose.keys())[:-1]:
                            x0, y0 = i.split('-')
                            x0, y0 = float(x0), float(y0)
                            x0l, x0r, y0l, y0r = (x0 - self.r) * scale + bbox[0], (x0 + self.r) * scale + bbox[0], (
                                    y0 - self.r) * scale + bbox[1], (y0 + self.r) * scale + bbox[1]
                            self.p_nochose.update({i: {
                                'index': self.canvas.create_oval(x0l, y0l, x0r, y0r, fill='green', activefill='red'),
                                'id': self.p_chose[i]['id']}})
                            self.canvas.delete(self.p_chose[i]['index'])
                            self.p_chose.pop(i)
                    break
            for x0,y0,x1,y1 in self.parent.add_line:
                if (x1-x0)==0:
                    yy=sorted([y0,y1])
                    if (y_<yy[0]) or (y_>yy[1]):
                        continue
                    xx=[x0-3,x0+3]
                    if (x_<xx[0])or (x_>xx[1]):
                        continue
                if (y1-y0)==0:
                    xx=sorted([x0,x1])
                    if (x_<xx[0]) or (x_>xx[1]):
                        continue
                    yy=[y0-3,y0+3]
                    if (y_<yy[0])or (y_>yy[1]):
                        continue
                if ((x1-x0)!=0)and((y1-y0)!=0):
                    ang=-np.arctan((y1-y0)/(x1-x0))
                    x1_,y1_=self.__rotate_coordinates(x1,y1,x0,y0,ang)
                    xx_,yy_=self.__rotate_coordinates(x_,y_,x0,y0,ang)
                    if x0<x1:
                        if (xx_<x0) or (xx_>x1_):
                            continue
                    else:
                        if (xx_>x0) or (xx_<x1_):
                            continue
                    yy=[y0-3,y0+3]
                    if (yy_<yy[0])or (yy_>yy[1]):
                        continue
                self.canvas.delete(self.add_chose[str(x0) + str(y0) + str(x1) + str(y1)]['index'])
                self.add_chose.pop(str(x0) + str(y0) + str(x1) + str(y1))
                self.parent.add_line.remove([x0, y0, x1, y1])
        else:
            key=list(self.parent.clear_rectangle.keys())
            for l in key:
                x0, y0, x1, y1=self.parent.clear_rectangle[l]['coordinates']
                minx, maxx = sorted([x0, x1])
                miny, maxy = sorted([y0, y1])
                jd=False
                if (x_>minx)and(x_<maxx)and(y_>miny-self.r)and(y_<miny+self.r):
                    jd=True
                if (x_>minx)and(x_<maxx)and(y_>maxy-self.r)and(y_<maxy+self.r):
                    jd=True
                if (x_>minx-self.r)and(x_<minx+self.r)and(y_>miny)and(y_<maxy):
                    jd=True
                if (x_>maxx-self.r)and(x_<maxx+self.r)and(y_>miny)and(y_<maxy):
                    jd=True
                if jd:
                    for i in range(4):
                        self.canvas.delete(self.add_rectangle[str(x0) + str(y0) + str(x1) + str(y1)]['index'][i])
                    lines=self.parent.clear_rectangle[l]['lines']
                    for li in lines:
                        if li in self.parent.removed_lines:
                            x0_, y0_, x1_, y1_=li
                            x0__, x1__, y0__, y1__ = x0_ * scale + bbox[0], x1_ * scale + bbox[0], y0_ * scale + bbox[
                                1], y1_ * scale + bbox[1]
                            self.nochose.update({str(x0_) + str(y0_) + str(x1_) + str(y1_): {
                                'index': self.canvas.create_line(x0__, y0__, x1__, y1__, width=2*self.r, fill='yellow',
                                                                 activefill='purple'),
                                'line': self.chose[str(x0_) + str(y0_) + str(x1_) + str(y1_)]['line']}})
                            self.canvas.delete(self.chose[str(x0_) + str(y0_) + str(x1_) + str(y1_)]['index'])
                            self.chose.pop(str(x0_) + str(y0_) + str(x1_) + str(y1_))
                            self.parent.removed_lines.remove([x0_, y0_, x1_, y1_])
                            self.parent.exist_lines.append([x0_, y0_, x1_, y1_])
                    self.add_rectangle.pop(str(x0) + str(y0) + str(x1) + str(y1))
                    self.parent.clear_rectangle.pop(str(x0) + str(y0) + str(x1) + str(y1))
            for x0,y0,x1,y1 in self.parent.lines:
                if (x1-x0)==0:
                    yy=sorted([y0,y1])
                    if (y_<yy[0]) or (y_>yy[1]):
                        continue
                    xx=[x0-1,x0+1]
                    if (x_<xx[0])or (x_>xx[1]):
                        continue
                if (y1-y0)==0:
                    xx=sorted([x0,x1])
                    if (x_<xx[0]) or (x_>xx[1]):
                        continue
                    yy=[y0-1,y0+1]
                    if (y_<yy[0])or (y_>yy[1]):
                        continue
                if ((x1-x0)!=0)and((y1-y0)!=0):
                    ang=-np.arctan((y1-y0)/(x1-x0))
                    x1_,y1_=self.__rotate_coordinates(x1,y1,x0,y0,ang)
                    xx_,yy_=self.__rotate_coordinates(x_,y_,x0,y0,ang)
                    if (xx_<x0) or (xx_>x1_):
                        continue
                    yy=[y0-1,y0+1]
                    if (yy_<yy[0])or (yy_>yy[1]):
                        continue
                x0_, x1_, y0_, y1_ = x0 * scale + bbox[0], x1 * scale + bbox[0], y0 * scale + bbox[1], y1 * scale + bbox[1]
                if str(x0) + str(y0) + str(x1) + str(y1) in self.nochose:
                    self.chose.update({str(x0) + str(y0) + str(x1) + str(y1): {
                        'index': self.canvas.create_line(x0_, y0_, x1_, y1_, width=self.r, fill='purple', activefill='yellow'),
                        'line': self.nochose[str(x0) + str(y0) + str(x1) + str(y1)]['line']}})
                    self.canvas.delete(self.nochose[str(x0) + str(y0) + str(x1) + str(y1)]['index'])
                    self.nochose.pop(str(x0) + str(y0) + str(x1) + str(y1))
                    self.parent.exist_lines.remove([x0,y0,x1,y1])
                    self.parent.removed_lines.append([x0,y0,x1,y1])
                else:
                    self.nochose.update({str(x0) + str(y0) + str(x1) + str(y1): {
                        'index': self.canvas.create_line(x0_, y0_, x1_, y1_, width=2*self.r, fill='yellow', activefill='purple'),
                        'line': self.chose[str(x0) + str(y0) + str(x1) + str(y1)]['line']}})
                    self.canvas.delete(self.chose[str(x0) + str(y0) + str(x1) + str(y1)]['index'])
                    self.chose.pop(str(x0) + str(y0) + str(x1) + str(y1))
                    self.parent.removed_lines.remove([x0, y0, x1, y1])
                    self.parent.exist_lines.append([x0, y0, x1, y1])
        self.parent.plot()
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

def lines_detect(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel_size = 5
    blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
    edges = cv2.Canny(blur_gray, 0, 0)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 20, np.array([]),
                            minLineLength=2, maxLineGap=5)
    return lines.squeeze()

class Drawing_window(ttk.Frame):
    def __init__(self, mainframe,parent,w,h):
        ttk.Frame.__init__(self, master=mainframe)
        self.master.title('Remove boundaries')
        self.master.geometry('1200x600')  # size of the main window
        self.master.rowconfigure(0, weight=1)  # make the CanvasImage widget expandable
        self.master.columnconfigure(0, weight=1)

        canvas = Draw_window(self.master,parent,w,h)  # create widget
        canvas.grid(row=0, column=0)  # show widget

class Adding_window(ttk.Frame):
    def __init__(self, mainframe,parent,w,h):
        ttk.Frame.__init__(self, master=mainframe)
        self.master.title('Adding boundaries and define destinations')
        self.master.geometry('1200x600')  # size of the main window
        self.master.rowconfigure(0, weight=1)  # make the CanvasImage widget expandable
        self.master.columnconfigure(0, weight=1)

        canvas = Draw_window(self.master,parent,w,h,removing=False)  # create widget
        canvas.grid(row=0, column=0)  # show widget

class Main_window(ttk.Frame):
    def __init__(self, master,knames,pts,work_path,Place,building,floor):
        ttk.Frame.__init__(self, master=master)
        self.knames=knames
        self.pts=pts

        self.work_path=work_path
        self.Place=Place
        self.building=building
        self.floor=floor
        self.outf=os.path.join(work_path,Place,building,floor, 'boundaries.json')
        abtn = tk.Button(self, text='Select Floor Plan', width=16, command=self.FloorPlan_select)
        abtn.grid(row=23, column=0, padx=5,pady=20, columnspan=1, rowspan=1)

        # self.destination={}

        windowWidth = self.master.winfo_reqwidth()
        windowHeight = self.master.winfo_reqheight()
        self.positionRight = int(self.master.winfo_screenwidth() / 2 - windowWidth / 2)
        self.positionDown = int(self.master.winfo_screenheight() / 2 - windowHeight / 2)
        self.master.geometry("+{}+{}".format(self.positionRight, self.positionDown))

        self.master.title('Boundary select')
        self.pack(side="left", fill="both", expand=False)
        self.master.geometry('2000x1200')
        self.master.columnconfigure(1, weight=1)
        self.master.columnconfigure(3, pad=7)
        self.master.rowconfigure(3, weight=1)
        self.master.rowconfigure(6, pad=7)
        #--------------------------------------------------


        #---------------------------------------------------
        separatorh = ttk.Separator(self, orient='horizontal')
        separatorh.grid(row=20, column=0,pady=10,ipadx=1,columnspan=3, rowspan=1,sticky="ew")
        separatorv1 = ttk.Separator(self, orient='vertical')
        separatorv1.grid(row=0, column=3, padx=10, columnspan=1, rowspan=70, sticky="sn")

        fbtn = tk.Button(self, text='Save data', width=16, command=self.save)
        fbtn.grid(row=24, column=0, padx=5,pady=20, columnspan=1, rowspan=1)
        gbtn = tk.Button(self, text='Reset boundaries', width=16, command=self.reset)
        gbtn.grid(row=25, column=0, padx=5, pady=20, columnspan=1, rowspan=1)
        hbtn = tk.Button(self, text='Remove add-on', width=16, command=self.remove)
        hbtn.grid(row=26, column=0, padx=5, pady=20, columnspan=1, rowspan=1)
        ebtn = tk.Button(self, text='Help', width=16, command=self.help)
        ebtn.grid(row=27, column=0, padx=5,pady=20, columnspan=1, rowspan=1)

    def FloorPlan_select(self):
        fl_selected = filedialog.askopenfilename(initialdir=opt.plan, title='Select Floor Plan')
        self.plan = cv2.imread(fl_selected)
        self.h, self.w, _ = self.plan.shape
        self.lines = lines_detect(self.plan)
        if os.path.exists(self.outf):
            with open(self.outf,'r') as f:
                data=json.load(f)
            self.exist_lines=data['lines']
            self.removed_lines=data['removed_lines']
            self.add_line=data['add_lines']
            self.clear_rectangle=data['clear_rectangle']
            # self.destination=data['destination']
            self.destination = {}
        else:
            self.exist_lines=self.lines
            self.removed_lines=[]
            self.add_line=[]
            self.destination = {}
            self.clear_rectangle={}
        self.plot()

    def plot(self):
        try:
            self.li.destroy()
        except:
            pass
        self.li = ttk.LabelFrame(self, text='Information')
        self.li.grid(row=0, column=0, rowspan=4,columnspan=2, sticky=W, pady=4, ipadx=2)
        self.lbl = Label(self.li, text="Exist boundaries:\t%d" % len(self.exist_lines))
        self.lbl.pack(side=TOP,padx=20,pady=10)
        self.lbl = Label(self.li, text="Removed boundaries:\t%d" % len(self.removed_lines))
        self.lbl.pack(side=TOP,padx=20,pady=5)
        self.lbl = Label(self.li, text="Added boundaries:\t%d" % len(self.add_line))
        self.lbl.pack(side=TOP,padx=20,pady=5)
        self.lbl = Label(self.li, text="Added destinations:\t%d" % len(self.destination))
        self.lbl.pack(side=TOP, padx=20, pady=5)
        im = np.ones((self.h, self.w, 3), dtype=np.uint8) * 255
        for line in self.exist_lines:
            x1, y1, x2, y2 = line
            cv2.line(im, (x1, y1), (x2, y2), (0, 0, 0), 1)
        for line in self.add_line:
            x1, y1, x2, y2 = line
            cv2.line(im, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 1)
        for k in self.destination.keys():
            x_, y_ = self.pts[self.knames.index(self.destination[k]['id'])]
            cv2.putText(im, self.destination[k]['name'], (x_-30, y_+50), cv2.FONT_HERSHEY_SIMPLEX,
                                  1, (0, 0, 255), 2, cv2.LINE_AA)
        im = Image.fromarray(im[:, :, ::-1])
        draw=ImageDraw.Draw(im)
        for i in range(len(self.knames)):
            x_, y_ = self.pts[i]
            draw.ellipse((x_ - 2, y_ - 2, x_ + 2, y_ + 2), fill=(0, 255, 0))
        for k in self.destination.keys():
            x_, y_ = self.pts[self.knames.index(self.destination[k]['id'])]
            vertices = self.star_vertices([x_, y_], 30)
            draw.polygon(vertices, fill='red')
        scale = 1600 / self.w
        newsize = (1600, int(self.h * scale))
        im = im.resize(newsize)
        tkimage1 = ImageTk.PhotoImage(im)
        self.myvar1 = Label(self, image=tkimage1)
        self.myvar1.bind('<Return>', self.new_window)
        self.myvar1.bind('<Double-Button-1>', self.new_window)
        self.myvar1.bind('<Double-Button-3>', self.add_window)
        self.myvar1.image = tkimage1
        self.myvar1.grid(row=0, column=4, columnspan=1, rowspan=40, sticky="snew")

    def help(self):
        self.info=tk.Toplevel(self.master)
        self.info.geometry('800x650')
        self.info.title('Instruction')
        self.info.geometry("+{}+{}".format(self.positionRight-300, self.positionDown-200))
        # This will create a LabelFrame
        label_frame1 = LabelFrame(self.info,height=100, text='Steps')
        label_frame1.pack(expand='yes', fill='both')

        label1 = Label(label_frame1, text='1. Choose a frame in frame list.')
        label1.place(x=0, y=5)

        label2 = Label(label_frame1, text='2. Double click or press <Enter> to open frame and pick a feature point.')
        label2.place(x=0, y=35)

        label3 = Label(label_frame1, text='3. Double click floor plan and choose corresponding point of previous feature point.')
        label3.place(x=0, y=65)

        label4 = Label(label_frame1, text='4. Repeat above until get good matching, save the topo-map.')
        label4.place(x=0, y=95)

        label_frame1 = LabelFrame(self.info,height=400, text='Buttons')
        label_frame1.pack(expand='yes', fill='both',side='bottom')

        label_1 = LabelFrame(label_frame1, height=60, text='Save Animation')
        label_1.place(x=5, y=23)
        label1 = Label(label_1, text='Save a Gif animation of the whole mapping trajectory')
        label1.pack()

        label_2 = LabelFrame(label_frame1, height=40, text='Select Floor Plan')
        label_2.place(x=5, y=70)
        label2 = Label(label_2, text='Select a floor plan of your project.')
        label2.pack()

        label_3 = LabelFrame(label_frame1, height=40, text='Delete last pair')
        label_3.place(x=5, y=117)
        label3 = Label(label_3,
                       text='Permanently Delete last pairs that used to calculate transformation matrix.')
        label3.pack()

        label_4 = LabelFrame(label_frame1, height=60, text='Left\Right button')
        label_4.place(x=5, y=164)
        label4 = Label(label_4, text='Left button moves the last current pair which used to compute matrix to trash,\nright button recovers fisrt pair in trash.')
        label4.pack()

        label_5 = LabelFrame(label_frame1, height=40, text='Delete all pairs')
        label_5.place(x=5, y=229)
        label5 = Label(label_5,
                       text='Permanently Delete all pairs in current and in trash.')
        label5.pack()

        label_6 = LabelFrame(label_frame1, height=40, text='Delete trash pairs')
        label_6.place(x=5, y=276)
        label6 = Label(label_6,
                       text='Permanently Delete all pairs in trash.')
        label6.pack()

        label_7 = LabelFrame(label_frame1, height=60, text='Save Topo-Map\Matrix')
        label_7.place(x=5, y=323)
        label7 = Label(label_7,
                       text='Save all current pairs, transformation matrix which will be load and reuse when reopen\nthe project. Save Topo-map which is used in localization.')
        label7.pack()

        label_8 = LabelFrame(label_frame1, height=40, text='Animation')
        label_8.place(x=5, y=388)
        label8 = Label(label_8,
                       text='Create an animation gif of mapping.')
        label8.pack()

    def star_vertices(self,center,r):
        out_vertex = [(r * np.cos(2 * np.pi * k / 5 + np.pi / 2- np.pi / 5) + center[0],
                       r * np.sin(2 * np.pi * k / 5 + np.pi / 2- np.pi / 5) + center[1]) for k in range(5)]
        r = r/2
        in_vertex = [(r * np.cos(2 * np.pi * k / 5 + np.pi / 2 ) + center[0],
                      r * np.sin(2 * np.pi * k / 5 + np.pi / 2 ) + center[1]) for k in range(5)]
        vertices = []
        for i in range(5):
            vertices.append(out_vertex[i])
            vertices.append(in_vertex[i])
        vertices = tuple(vertices)
        return vertices

    def Remove(self):
        self.add_line = []
        self.plot()
        self.win.destroy()
        showinfo("Removed!", "Removed all add-on boundaries")

    def remove(self):
        self.win = tk.Toplevel(self.master)
        self.win.wm_title("Warning!!!")
        self.win.geometry('330x130')
        self.win.geometry("+{}+{}".format(self.positionRight, self.positionDown))
        l = tk.Label(self.win, text="Removed all add-on boundaries?")
        l.grid(row=1, column=0, columnspan=4, rowspan=2,
               padx=20, pady=30)

        b1 = ttk.Button(self.win, text="Yes", width=6, command=self.Remove)
        b1.grid(row=3, column=0, columnspan=1, padx=40, rowspan=3)

        b2 = ttk.Button(self.win, text="No", width=6, command=self.win.destroy)
        b2.grid(row=3, column=1, columnspan=1, padx=40, rowspan=3)

    def Reset(self):
        self.exist_lines=self.lines
        self.removed_lines=[]
        self.add_line =[]
        self.clear_rectangle = {}
        self.plot()
        self.win.destroy()
        showinfo("Reset!", "All boundaries recovered")

    def reset(self):
        self.win = tk.Toplevel(self.master)
        self.win.wm_title("Warning!!!")
        self.win.geometry('330x130')
        self.win.geometry("+{}+{}".format(self.positionRight, self.positionDown))
        l = tk.Label(self.win, text="Reset all boundaries?")
        l.grid(row=1, column=0, columnspan=4, rowspan=2,
               padx=20, pady=30)

        b1 = ttk.Button(self.win, text="Yes",width=6, command=self.Reset)
        b1.grid(row=3, column=0, columnspan=1,padx=40, rowspan=3)

        b2 = ttk.Button(self.win, text="No",width=6, command=self.win.destroy)
        b2.grid(row=3, column=1, columnspan=1,padx=40, rowspan=3)

    def save_data(self):
        destinations=os.path.join(self.work_path,'destination.json')
        new_dest=[]
        new_floor= {}
        for k,v in self.destination.items():
            name=v['name']
            new_dest.append({name:v['id']})
        new_floor.update({self.floor:new_dest})
        new_floor = dict(collections.OrderedDict(sorted(new_floor.items())))
        if not os.path.exists(destinations):
            list={self.Place:{self.building:new_floor}}
        else:
            with open(destinations,'r') as f:
                list=json.load(f)
            if self.Place in list:
                if self.building in list[self.Place]:
                    building=list[self.Place][self.building]
                    building.update({self.floor: new_dest})
                    building=collections.OrderedDict(sorted(building.items()))
                    list[self.Place].update({self.building:dict(building)})
                else:
                    Place=list[self.Place]
                    Place.update({self.building: new_floor})
                    collections.OrderedDict(sorted(Place.items()))
                    list.update({self.Place:dict(Place)})
            else:
                Place={self.Place:{self.building:new_floor}}
                collections.OrderedDict(sorted(Place.items()))
                list.update(dict(Place))
        with open(destinations,'w') as f:
            json.dump(list,f)
        with open(self.outf,'w') as f:
            json.dump({'lines':self.exist_lines,'add_lines':self.add_line,'removed_lines':self.removed_lines,'clear_rectangle':self.clear_rectangle,'destination':self.destination},f)
        self.win.destroy()
        showinfo("Saved!", "Boundaries has been saved")

    def save(self):
        self.win = tk.Toplevel(self.master)
        self.win.wm_title("Warning!!!")
        self.win.geometry('330x130')
        self.win.geometry("+{}+{}".format(self.positionRight, self.positionDown))
        l = tk.Label(self.win, text="Save boundaries?")
        l.grid(row=1, column=0, columnspan=4, rowspan=2,
               padx=20, pady=30)

        b1 = ttk.Button(self.win, text="Yes",width=6, command=self.save_data)
        b1.grid(row=3, column=0, columnspan=1,padx=40, rowspan=3)

        b2 = ttk.Button(self.win, text="No",width=6, command=self.win.destroy)
        b2.grid(row=3, column=1, columnspan=1,padx=40, rowspan=3)

    def remove_confirm(self):
        self.destination.pop(self.index)
        self.plot()
        self.win.destroy()

    def remove_destination(self,index):
        self.index = index
        self.win = tk.Toplevel(self.master)
        self.win.wm_title("Remove destination")
        self.win.geometry('330x100')
        self.win.geometry("+{}+{}".format(self.positionRight, self.positionDown))
        self.shift = tk.Frame(self.win)
        self.shift.grid(row=0, column=0)
        self.shift1 = tk.Frame(self.shift)
        self.shift1.grid(row=0, column=0, pady=10, padx=60)
        scale = Label(self.shift1, text='Delete this destination?')
        scale.pack(side=LEFT)
        b1 = ttk.Button(self.win, text="Confirm", width=8, command=self.remove_confirm)
        b1.grid(row=4, column=0, columnspan=1, padx=30, pady=10, rowspan=1, sticky='w')
        b2 = ttk.Button(self.win, text="Cancel", width=8, command=self.win.destroy)
        b2.grid(row=4, column=0, columnspan=1, padx=20, pady=10, rowspan=1, sticky='e')

    def name_back(self):
        self.destination[self.index].update({'name':str(self.e4.get())})
        self.plot()
        self.win.destroy()

    def name_destination(self,index):
        self.index=index
        self.win = tk.Toplevel(self.master)
        self.win.wm_title("Define destination name")
        self.win.geometry('550x100')
        self.win.geometry("+{}+{}".format(self.positionRight, self.positionDown))
        self.shift = tk.Frame(self.win)
        self.shift.grid(row=0, column=0)
        self.shift1 = tk.Frame(self.shift)
        self.shift1.grid(row=0, column=0, pady=10, padx=60)
        scale = Label(self.shift1, text='Name of this destination:')
        scale.pack(side=LEFT)
        self.e4 = tk.Entry(self.shift1, width=20, justify='center')
        self.e4.pack(side=LEFT)
        b1 = ttk.Button(self.win, text="Confirm", width=8, command=self.name_back)
        b1.grid(row=4, column=0, columnspan=1, padx=30,pady=10, rowspan=1, sticky='w')
        b2 = ttk.Button(self.win, text="Cancel", width=8, command=self.win.destroy)
        b2.grid(row=4, column=0, columnspan=1, padx=20,pady=10, rowspan=1, sticky='e')

    def new_window(self,w):
        self.newWindow = tk.Toplevel(self.master)
        self.app1 = Drawing_window(self.newWindow,self,self.w,self.h)

    def add_window(self,w):
        self.newWindow = tk.Toplevel(self.master)
        self.app1 = Adding_window(self.newWindow,self,self.w,self.h)

def main(opt):
    knames,pts= load_data(os.path.join(opt.work_path,opt.Place,opt.Building,opt.Floor))
    style = Style(theme='superhero')
    root = style.master
    Main_window(root, knames, pts, opt.work_path, opt.Place, opt.Building, opt.Floor)
    root.mainloop()

if __name__ == '__main__':
    opt = options()
    main(opt)
