import numpy as np
import cv2 as cv
import glob
import os
from tqdm import tqdm
# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((7*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:7].T.reshape(-1,2)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
images = glob.glob('/home/endeleze/Desktop/WeNev/Mapping/data/src_images/calibration/*.png')
for fname in tqdm(images):
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (9,7), None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)
        # Draw and display the corners

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
print(mtx,'/n',dist)
# root='/home/endeleze/Desktop/WeNev/Mapping/data/src_images'
# src='GA_q'
# des='GA_q_undistort'
#
# src_path=os.path.join(root,src)
# # des_path=os.path.join(root,des)
# # if not os.path.exists(des_path):
# #     os.makedirs(des_path)
# dics=os.listdir(src_path)
#
# for dic in tqdm(dics):
#     img = cv.imread(os.path.join(src_path,dic))
#     h,  w = img.shape[:2]
#     newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
#
#     dst = cv.undistort(img, mtx, dist, None, newcameramtx)
#     # crop the image
#     x, y, w, h = roi
#     dst = dst[y:y+h, x:x+w]
#     cv.imwrite(os.path.join(src_path,dic), dst)