'''
Copyright 2022 Vignesh(VK)Kotteeswaran <iamvk888@gmail.com>
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import cv2
import detector_inference
import numpy as np
from glob import glob
from yolo2rectangle import rect2yolo

doc_client=detector_inference.Detection(model_path='models/document_detector.h5',
                                                classes_path='classes/document_classes.txt',
                                                grid=(16,16))

anglebracket_client=detector_inference.Detection(model_path='models/anglebracket_detector.h5',
                                                classes_path='classes/ab_classes.txt',
                                                grid=(64,64))

doc_client.plot_boxes=True
#aug=albu.Rotate(border_mode=cv2.BORDER_CONSTANT)

files=sorted(glob('/home/vk/personal/proj/scanbot-test-task-vignesh-kotteeswaran-main/data/ab_full/train/*.*g'))


def write(file,text):
    with open(file,'a')as f:
        f.write(text)

for file in files:

    ext=file.split('/')[-1].split('.')[-1]

    txt_file=file.replace(ext,'txt')
    if os.path.exists(txt_file):
        os.remove(txt_file)

    #print('file:',file)

    if True:

        img=cv2.imread(file)
        
        src_img=img.copy()
        mrz_img=img.copy()
        result_img=img.copy()
        masked_img=np.zeros((img.shape[0],img.shape[1]))
        angle=0


        #predict document RoI coordinates
        
        doc_result=doc_client.detect(img)
        
        doc_boxes=doc_result['boxes']
        doc_classes=doc_result['classes']
        #print(doc_result)

        if len(doc_boxes)>0:

            for class_,box in zip(doc_classes,doc_boxes): 

                if class_==1:

                    
                    mrz_xmin,mrz_ymin,mrz_xmax,mrz_ymax=box
                    _,mrz_cx,mrz_cy,mrz_w,mrz_h=rect2yolo(np.array([[0.,mrz_xmin,mrz_ymin,mrz_xmax,mrz_ymax]]),width=img.shape[1],height=img.shape[0]).squeeze()
                    
                    write(txt_file,f'2 {mrz_cx} {mrz_cy} {mrz_w} {mrz_h}\n')

            
            for class_,box in zip(doc_classes,doc_boxes):

                

                if class_==0:


                    xmin,ymin,xmax,ymax=box
                    _,doc_cx,doc_cy,doc_w,doc_h=rect2yolo(np.array([[0.,xmin,ymin,xmax,ymax]]),width=img.shape[1],height=img.shape[0]).squeeze()
                    write(txt_file,f'1 {doc_cx} {doc_cy} {doc_w} {doc_h}\n')
                    

                    xmin,ymin,xmax,ymax=int(xmin),int(ymin),int(xmax),int(ymax)
                    result_img=img.copy()
                    mrz_img=img.copy()
                    
                    
                    
                    #extract document RoI
                    ab_roi=img[ymin:ymax,xmin:xmax]

                    #predict "<" coordinates   
                    ab_result=anglebracket_client.detect(ab_roi)
                    ab_boxes=ab_result['boxes']


                    
                    for ab_box in ab_boxes:
                        ab_xmin,ab_ymin,ab_xmax,ab_ymax=ab_box
                        _,ab_cx,ab_cy,ab_w,ab_h=rect2yolo(np.array([[0.,xmin+ab_xmin,ymin+ab_ymin,xmin+ab_xmax,ymin+ab_ymax]]),width=img.shape[1],height=img.shape[0]).squeeze()
                        write(txt_file,f'0 {ab_cx} {ab_cy} {ab_w} {ab_h}\n')
                            
                       




