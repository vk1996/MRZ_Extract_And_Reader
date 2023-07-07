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
from time import time
from utils import recover_mrz,mask,find_skew_angle,deskew,return_highest_ymax,return_lowest_ymin,return_tallest_area,return_thickest_area,check_multiple_line,return_farthest_xmax,return_nearest_xmin



doc_client=detector_inference.Detection(model_path='models/document_detector.h5',
                                                classes_path='classes/document_classes.txt',
                                                grid=(16,16))

anglebracket_client=detector_inference.Detection(model_path='models/anglebracket_detector.h5',
                                                classes_path='classes/ab_classes.txt',
                                                grid=(64,64))
doc_client.plot_boxes=True

files=sorted(glob('test/qc/*.*g'))

for file in files:

    start_time=time()

    print('file:',file)

    try:

        img=cv2.imread(file)
        
        src_img=img.copy()
        mrz_img=img.copy()
        result_img=img.copy()
        masked_img=np.zeros((img.shape[0],img.shape[1]))


        #predict document RoI coordinates
        tic=time()
        doc_result=doc_client.detect(img)
        print('Time taken for document detection:',time()-tic)
        doc_boxes=doc_result['boxes']
        doc_classes=doc_result['classes']
        #print(doc_result)

        if len(doc_boxes)>0:
            
            for class_,box in zip(doc_classes,doc_boxes):

                

                if class_==0:


                    xmin,ymin,xmax,ymax=box

                    # xmin=max(0,xmin-0.05*img.shape[1])
                    # xmax=min(xmax,xmax+0.05*img.shape[1])
                    

                    #find skew angle of document RoI and deskew the image
                    angle=find_skew_angle(img[ymin:ymax,xmin:xmax])
                    print('Angle:',angle)
                    if 0<angle<=45 or angle>=-45:
                        img=deskew(img,angle)
                    result_img=img.copy()
                    mrz_img=img.copy()
                    
                    
                    #extract document RoI
                    ab_roi=img[ymin:ymax,xmin:xmax]

                    #predict "<" coordinates   
                    tic=time()
                    ab_result=anglebracket_client.detect(ab_roi)
                    print('Time taken for < detection:',time()-tic)
                    ab_boxes=ab_result['boxes']


                    if len(ab_boxes)>0:

                        #create "<" mask from "<" box coordinates
                        masked_img[ymin:ymax,xmin:xmax]=mask(ab_roi,ab_boxes)


                        #compute "<" box attributes like maximum thickness ,height among the boxes
                        #nearest xmin,ymin farthest xmax,ymax of "<" boxes to estimate the area of MRZ

                        thickness=return_thickest_area(ab_boxes)
                        height=return_tallest_area(ab_boxes)
                        
                        
                        lowest_ymin=ymin+return_lowest_ymin(ab_boxes)
                        nearest_xmin=xmin+return_nearest_xmin(ab_boxes)
                        
                        #logic to overcome uncertainity of "<" existense of second char of 
                        #first row of MRZ
                        if return_nearest_xmin(ab_boxes)>0.15*ab_roi.shape[1]:
                            nearest_xmin=max(xmin,nearest_xmin-(0.3*img.shape[1]))
                        farthest_xmax=xmin+return_farthest_xmax(ab_boxes)+2*thickness

                        
                        #MRZ height can be computed as MRZ standard has maximum 2-3 rows
                        #similarly MRZ has nearly 44 chars in a row 
                        #Calculate MRZ width by box thickness times 42 from the rightmost "<" box
                        mrz_xmin=int(max(xmin+thickness,min(nearest_xmin-2*thickness,farthest_xmax-(42*thickness))))
                        mrz_xmax=int(min(xmax,farthest_xmax))
                        mrz_ymin=int(lowest_ymin-(0.5*height))

                        #check if "<" is available in more than one MRZ row
                        if check_multiple_line(ab_boxes):
                            mrz_ymax=int(ymin+return_highest_ymax(ab_boxes)+(height))
                        else:
                            mrz_ymax=int(lowest_ymin+(5*height))

                        #recover selected MRZ RoI
                        mrz_img[mrz_ymin:mrz_ymax,mrz_xmin:mrz_xmax]=recover_mrz(mrz_img[mrz_ymin:mrz_ymax,mrz_xmin:mrz_xmax])


                    for ab_box in ab_boxes:
                            
                            cv2.rectangle(result_img, (int(xmin+ab_box[0]),int(ymin+ab_box[1])), (int(xmin+ab_box[2]),int(ymin+ab_box[3])),(0,0,255), 2)
                        
                    

                
                    

    
               
                
                

    except Exception as e:
        print('[Error]:',e)

    print('Process time:',time()-start_time)
            
    
    cv2.imshow('input',doc_client.show_frame)
    cv2.imshow('<',anglebracket_client.show_frame)
    cv2.imshow('mask',masked_img)
    cv2.imshow('deskewed_intermediate',result_img)
    cv2.imshow('output',mrz_img)
    k=cv2.waitKey(0)

    if k==ord('q'):
        cv2.destroyAllWindows()
        break


