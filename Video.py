# -*- coding: utf-8 -*-
"""
Created on Wed May  8 16:49:27 2019

@author: balam
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
from UndistortImage import UndistortImage
from ReadCameraModel import ReadCameraModel
import math
import glob
import copy

car = glob.glob("Oxford_dataset/stereo/centre/*.png")
car.sort()

carImage_a = cv2.imread(car[0], cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH)

#video = cv2.VideoWriter('video.avi',cv2.VideoWriter_fourcc(*"XVID"), 10,
#                        (carImage_a.shape[0],carImage_a.shape[1]))
video = cv2.VideoWriter('video.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 50, (carImage_a.shape[1],carImage_a.shape[0]))

fx, fy, cx, cy, G_camera_image, LUT = ReadCameraModel('Oxford_dataset/model')
K = np.array([[fx, 0, cx],[0, fy, cy],[0, 0, 1]]) 
count =0

for path in car:
    count +=1
    print(count)
    carImage_a = cv2.imread(path, cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH)
    carImage_a = cv2.cvtColor(carImage_a, cv2.COLOR_BAYER_GR2BGR)
    carImage_a = UndistortImage(carImage_a, LUT)
    cv2.imshow("rest",carImage_a)
    video.write(carImage_a)
    cv2.waitKey(1)

video.release()
    