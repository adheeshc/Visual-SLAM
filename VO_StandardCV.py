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

def get3dPoint(pt,K):
    imgPt = np.hstack((pt,np.ones((pt.shape[0],1))))
    imgPt = imgPt.T
    R = np.eye(3)
    t = np.array([[0],[1.],[0]])
    P = K.dot(np.hstack((R,t)))
#    P = np.hstack((K,np.array([0,0,0]).reshape(3,1)))
    pInv = np.linalg.pinv(P)
    pt3d = np.dot(pInv, imgPt)
    return pt3d/ pt3d[3,:]

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
    tempImg = np.copy(frame2[:800,:])
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

    for pt in x2:
        cv2.circle(tempImg, tuple(pt), 5, (0,255,0),-1)
    
    # Essential matrix
    E,mask = cv2.findEssentialMat(x2, x1, K, cv2.FM_RANSAC, 0.999, 1.0, None)
    
    # R and T
    _, R, t, mask = cv2.recoverPose(E, x2, x1, K)
    

    
    
    current_pos += current_rot.dot(t) 
    current_rot = R.dot(current_rot)

    x,y,z = current_pos[0,0],current_pos[1,0],current_pos[2,0]
    xPlot.append(x)
    yPlot.append(y)
    zPlot.append(z)
    
    
    X2 = get3dPoint(x2,K)
    X2[3,:] *= -1 
    proj = X2[:3,:] + current_pos
    
    ax.scatter(x, y, z, marker='o', color = 'b', s = 40)
    ax.scatter(proj[0,:], proj[1,:], proj[2,:], marker='+', color = 'g', s = 20)
    if count % 10 == 0:
        plt.pause(0.001)
        plt.show()
    
    cv2.imshow("temp", tempImg)
    
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

ax.scatter(xPlot, yPlot, zPlot, marker='o', color = 'b')    

plt.show()
cap.release()
cv2.destroyAllWindows()

