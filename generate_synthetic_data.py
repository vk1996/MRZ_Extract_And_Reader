'''
Generate synthetic data by pasting cropped documents on
various daily life background to improve model robustness
'''
import numpy as np
from glob import glob
import os
import cv2
import random
import albumentations as A


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

pascal_files=sorted(glob('Enter background image path here'))

num_iterations=3

dir='data/document/test'


dest_dir='destination dir'
os.system('rm -r '+dest_dir+'/*')

files=sorted(glob(dir+'/*.txt'))

print('Num src files:',len(files))

img_ext='.jpg'



scale_factors=[i*0.1 for i in range(10,30,(30-10)//num_iterations)]
print('Scales:',scale_factors)

for iter in range(num_iterations):

    try:

        for file in files:

            

            img_file=file.replace('.txt',img_ext)

            bg_file=random.choice(pascal_files)
            bg_ext=bg_file.split('/')[-1].split('.')[-1]

            bg=cv2.imread(bg_file)



            data=np.loadtxt(file)

            if len(data.shape)<2:
                data=np.expand_dims(data,axis=0)
            classes=data[:,0]
            img=cv2.imread(img_file)

            scale_factor=scale_factors[iter]

            img=cv2.resize(img,(int(bg.shape[1]//scale_factor),int(bg.shape[0]//scale_factor)))

            #bg_h,bg_w=int(bg.shape[0]//6),int(bg.shape[1]//6)

            bg_h=max(int(np.random.uniform(low=1,high=bg.shape[0]-img.shape[0]-10)),0)
            bg_w=max(int(np.random.uniform(low=1,high=bg.shape[1]-img.shape[1]-10)),0)
            #print(img.shape,bg.shape)

            img_h,img_w=img.shape[0],img.shape[1]

            height=img.shape[0]
            width=img.shape[1]

            box=yolo2rect(data,height=height,width=width)
            

            
            #print(box.shape)


            box[:,1]=box[:,1]+bg_w
            box[:,3]=box[:,3]+bg_w
            box[:,2]=box[:,2]+bg_h
            box[:,4]=box[:,4]+bg_h
            if 0 not in classes:
                box=np.concatenate((box,np.array([[0,bg_w,bg_h,bg_w+img_w,bg_h+img_h]])),axis=0)

            


            box=rect2yolo(box.copy(),height=bg.shape[0],width=bg.shape[1])
            bg[bg_h:bg_h+img_h,bg_w:bg_w+img_w]=img


            bgname=dest_dir+'/'+file.split('/')[-1][:-4]+f'_{iter}_'+bg_file.split('/')[-1]


            cv2.imwrite(bgname,np.uint8(bg))

            for coords in box:
                with open(bgname.replace(bg_ext,'txt'),'a')as f:
                    class_index,cx,cy,w,h=coords[0:]
                    f.write(f'{int(class_index)} {cx} {cy} {w} {h} \n')

    except Exception as e:
        print(bg.shape,img.shape)
        print('[Error]:',e)
    

    
