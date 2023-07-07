import cv2
import numpy as np
from glob import glob

files=sorted(glob('/home/vk/personal/proj/scanbot-test-task-vignesh-kotteeswaran-main/data/ab_full/test/*.jpg'))

for file in files:

    img = cv2.imread(file)
    img=cv2.resize(img,(512,1024))
    src_img=img.copy()
    background = np.zeros(img.shape, dtype=np.uint8)
    masked_bg= np.zeros(img.shape, dtype=np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bilateral = cv2.bilateralFilter(gray, 5, 5,5)
    eq = cv2.equalizeHist(bilateral)
    edged = cv2.Canny(eq, 100, 255)
    # rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(0.003*img.shape[1]),(int(0.005*img.shape[0]))))
    # edged=cv2.morphologyEx(edged, cv2.MORPH_CLOSE, rectKernel)

    contours,_= cv2.findContours(edged, cv2.RETR_TREE,  cv2.CHAIN_APPROX_SIMPLE)
    h, w = img.shape[:2]
    thresh_area = 0.01
    list_contours = []
    areas=[]
    for c in contours:
        area = cv2.contourArea(c)

        if True:

        #if (area > thresh_area*h*w): 
            x,y,w,h = cv2.boundingRect(c)
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
            # areas.append(area)
            # xmin,ymin,xmax,ymax = np.int0(cv2.boxPoints(cv2.minAreaRect(c)))
            # print((xmin,ymin),(xmax,ymax))
            # # print(rect_page)
            # # box_page = np.int0(cv2.boxPoints(rect_page))
            # list_contours.append((int(xmin),int(ymin),int(xmax),int(ymax)))

    # xmin,ymin,xmax,ymax=list_contours[int(np.argmax(np.array(areas)))]
    # cv2.rectangle(img, (int(xmin),int(ymin)), (int(xmax),int(ymax)),(255,0,0), 2) 

    # if len(list_contours)>0:
    #     c = max(list_contours, key=cv2.contourArea)
    #     print(c)
    #     mask = np.zeros(img.shape, dtype=np.uint8)
    #     mask = cv2.fillPoly(img=src_img.copy(), pts=c.reshape(1, -2, 2), color=(0,0,0))
    #     masked_image = cv2.bitwise_and(img, ~mask)

        
    #     background[:,:,:] = 128
    #     masked_bg = cv2.bitwise_and(background, mask)

    cv2.imshow('input',src_img)
    cv2.imshow('edges',edged)
    cv2.imshow('output',img)
    k=cv2.waitKey(0)
    
    k=cv2.waitKey(0)
    if k==ord('q'):
        break

cv2.destroyAllWindows()
    