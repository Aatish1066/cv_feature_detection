import cv2
import numpy as np

img1 = cv2.imread('query/armas (2815).jpg.rf.21606d51c1b2060234e1f513b13f0462.jpg' ,0)
img2 = cv2.imread('train/armas (2650).jpg.rf.9024bbedc37607dece52305c3f6080b3.jpg' ,0)

orb = cv2.ORB_create(nfeatures=5000)
kp1 ,desc1 = orb.detectAndCompute(img1 ,None)
kp2 ,desc2 = orb.detectAndCompute(img2 ,None)

# imgkp1=cv2.drawKeypoints(img1,kp1,None)
# imgkp2=cv2.drawKeypoints(img2,kp2,None)
bf = cv2.BFMatcher()
matches = bf.knnMatch(desc1 ,desc2 ,k=2)
good = []
for m ,n in matches:
    if m.distance < 0.75 * n.distance:
        good.append([m])

im3 = cv2.drawMatchesKnn(img1 ,kp1 ,img2 ,kp2 ,good ,None ,flags=2)
# cv2.imshow('kp1',imgkp1)
# cv2.imshow('kp2',imgkp2)
# cv2.imshow('img1',img1)
# cv2.imshow('img2',img2)
cv2.imshow('img3' ,im3)

cv2.waitKey(0)