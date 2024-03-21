import pvporcupine
import struct
import pyaudio
import pvleopard
import shutil
import os
from tqdm import tqdm
from multiprocessing import Pool

root='/media/endeleze/Endeleze_5T/UNav/Mapping/data/src_images/New_York_City/LightHouse'
src_path=os.path.join(root,'3_','perspective_images')
des_path=os.path.join(root,'3','perspective_images')

im_list_src=os.listdir(src_path)
im_list_des=os.listdir(des_path)

im_list=list(set(im_list_src).difference(set(im_list_des)))

def _foo(im):
    path = os.path.join(src_path, im)
    shutil.copy(path, des_path)

with Pool(16) as p:
    r = list(tqdm(p.imap(_foo, im_list), total=len(im_list)))

