import cv2
import os
orb = cv2.ORB_create(nfeatures=5000)
path = 'query'
##import
images = []
classnames = []
mylist = os.listdir(path)
print('total classes detected ' ,len(mylist))

for cl in mylist:
    imgcur = cv2.imread(f'{path}/{cl}' ,0)
    images.append(imgcur)
    classnames.append(os.path.splitext(cl)[0])
##end

def finddes(images):
    deslist = []
    for img in images:
        kp ,desc = orb.detectAndCompute(img ,None)
        deslist.append(desc)
    return deslist

def findid(img,deslist,thresh=15):
    kp2 ,desc2 = orb.detectAndCompute(img ,None)
    bf = cv2.BFMatcher()
    matchlist=[]
    finval=-1
    try:
        for desc1 in deslist:
             matches = bf.knnMatch(desc1 ,desc2 ,k=2)
             good = []
             for m ,n in matches:
                 if m.distance < 0.84 * n.distance:
                     good.append([m])
             matchlist.append(len(good))
    except:
        pass
    if len(matchlist)!=0:
        if max(matchlist)>thresh:
            finval= matchlist.index(max(matchlist))
    return finval
deslist = finddes(images)

cap = cv2.VideoCapture(0)

while True:
    succ , img2 = cap.read()
    imgorg = img2.copy()
    img2=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    id= findid(img2,deslist)
    if id != -1:
        cv2.putText(imgorg,classnames[id],(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
    cv2.imshow('detect' ,imgorg)
    cv2.waitKey(1)

