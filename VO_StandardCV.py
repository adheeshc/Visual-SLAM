"""
Perception Project 5
Group 9
"""
import numpy as np
import cv2
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import random
from ReadCameraModel import ReadCameraModel
import copy

#plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('X- axis')
ax.set_ylabel('Y- axis')
ax.set_zlabel('Z- axis')

fx, fy, cx, cy, G_camera_image, LUT = ReadCameraModel('Oxford_dataset/model')
K = np.array([[fx, 0, cx],[0, fy, cy],[0, 0, 1]]) 

current_pos = np.zeros((3, 1))
current_rot = np.eye(3)


xPlot = []
yPlot = []
zPlot = []
origin = np.zeros((4,1))
origin[3][0]= 1
H = np.eye(4)
def computeH(R,t):
    t = t.reshape(3,1)
    h = np.hstack((R,t))
    h = np.vstack((h, np.array([0,0,0,1])))
    return h

count = 0
cap = cv2.VideoCapture('undistorted.avi')

ret, frame1 = cap.read()
frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
#frame1 = cv2.GaussianBlur(frame1,(3,3),0)
frame1 = frame1[:800,:]
frame1= cv2.equalizeHist(frame1)

detector = cv2.ORB_create(1500)

prevFrame = frame1
prevKp , prevDes = detector.detectAndCompute(frame1,None)



while(cap.isOpened()):
    count +=1
    if count%100==0:
        print(count)
    ret, frame2 = cap.read()
    if not ret:
        break
#    frame2 = cv2.GaussianBlur(frame2,(3,3),0)
    frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    frame2 = frame2[:800,:]
    frame2= cv2.equalizeHist(frame2)
    
    kp1, des1 = prevKp, prevDes
    kp2, des2 = detector.detectAndCompute(frame2,None)
    
    prevKp, prevDes = kp2, des2
    
    if count<30:
        continue
    
    # BF Matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
#    bf = cv2.BFMatcher()
    matches = bf.match(des1,des2)
    matches = sorted(matches, key = lambda x:x.distance)
    
    
    x1 = np.float32([kp1[match.queryIdx].pt for match in matches])
    x2 = np.float32([kp2[match.trainIdx].pt for match in matches])

    
    # Essential matrix
    E,mask = cv2.findEssentialMat(x2, x1, K, cv2.FM_RANSAC, 0.999, 1.0, None)
    
    # R and T
    _, R, t, mask = cv2.recoverPose(E, x2, x1, K)
    

    
    
    current_pos += current_rot.dot(t) 
    current_rot = R.dot(current_rot)

    x,y,z = current_pos[0,0],current_pos[1,0],current_pos[2,0]
    xPlot.append(x)
    yPlot.append(y)
    zPlot.append(-z)
    
    
    
#    ax.scatter(x, y, z, marker='o', color = 'b')

#    if count % 10 == 0:
#        plt.pause(0.001)
#        plt.show()
    
    
    
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

ax.scatter(xPlot, yPlot, zPlot, marker='o', color = 'b')    

plt.show()
cap.release()
cv2.destroyAllWindows()

