import math
import cv2
import numpy as np    
import os
import time
import itertools
import pickle
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
def rsz(img):
    width = 640
    height = int(img.shape[0]*(640/img.shape[1]))
    return cv2.resize(img, (width,height))

def sim_score(image1,image2):
    feature_detector = cv2.xfeatures2d.SIFT_create()
        
    kp1, des1 = feature_detector.detectAndCompute(image1,None)
    kp2, des2 = feature_detector.detectAndCompute(image2,None)
    matches = None
    
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    good = []
    pts1 = []
    pts2 = []

    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.9 * n.distance:
            good.append(m)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)
    
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.RANSAC)

    # We select only inlier points
    pts1 = pts1[mask.ravel()==1]
    pts2 = pts2[mask.ravel()==1]
    # print(len(pts1))
    if len(pts1) > 20:
        return len(pts2)
    return 0


res=open('dbow_processed.txt')
valid=0
total=0
for line in tqdm(res):
    imgs=[]
    j=0
    for s in line.strip().split():
        s1=s[16:]
        img=cv2.imread(s1)
        width = 640
        height = int(img.shape[0]*(640/img.shape[1]))
        img=cv2.resize(img, (width,height))
        imgs.append(img)
    score=[]
    s_base=sim_score(imgs[0],imgs[0])
    for i in range(1,6):
        sc=sim_score(imgs[0],imgs[i])
        sr=math.ceil(sc/float(s_base) * 1000.0) / 1000.0
        if (sr>=0.05):
            valid+=1
        total+=1
print('valid: '+str(valid))
print('total: '+str(total))