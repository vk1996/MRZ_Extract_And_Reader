import os
import cv2
from matplotlib import pyplot as plt
from glob import glob
import numpy as np

def yolo2rect(dpred,width=1.0,height=1.0):
    #0.923047 0.481250 0.086719 0.445833

    cx=dpred[:,1]*width
    cy=dpred[:,2]*height
    w=dpred[:,3]*width
    h=dpred[:,4]*height

    #print(x,y,w,h)

    xmin,ymin= cx-(w/2),cy-(h/2)
    xmax,ymax= cx+(w/2),cy+(h/2)

    box=np.copy(dpred)
    box[:,1]=xmin
    box[:,2]=ymin
    box[:,3]=xmax
    box[:,4]=ymax

    return box

def rect2yolo(dpred,width=1.0,height=1.0):
    xmin=dpred[:,1]
    ymin=dpred[:,2]
    xmax=dpred[:,3]
    ymax=dpred[:,4]

    h=(ymax-ymin)/height
    w=(xmax-xmin)/width
    cx=((xmin+xmax)/2)/width
    cy=((ymin+ymax)/2)/height

    yolobox=np.copy(dpred)
    yolobox[:,1]=cx
    yolobox[:,2]=cy
    yolobox[:,3]=w
    yolobox[:,4]=h

    return yolobox

