'''
Copyright 2023 Vignesh(VK)Kotteeswaran <iamvk888@gmail.com>
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


import cv2
import numpy as np
import onnxruntime as ort




class Detection:
    def __init__(
        self,
        model_path,
        classes_path,
        grid=(16,16),
        image_size=(512,512,3),
        conf_threshold=0.25,
        iou_threshold=0.1,
        show_model_layer=False,
        ):

        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.__img_size = image_size
        
    
        self.grid_w=grid[0]
        self.grid_h=grid[1]

        self.__model = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

        self.plot_boxes=False
        self.__bbox_color=(0,0,255)
        self.show_frame=None

        with open(classes_path,'r') as f:
            data=f.read()
        self.labels=[i for i in data.split('\n') if len(i)>0]


        

    def detect(self,src):

        result={}

        if isinstance(src,str):
            input_image = cv2.imread(src)
        else:
            input_image=src.copy()

        self.show_frame=input_image.copy()

        #preprocess the image


        blob =cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        blob =cv2.resize(blob , (self.__img_size[0], self.__img_size[1]),interpolation=cv2.INTER_AREA)
        blob =np.expand_dims(np.array(blob /255.,dtype=np.float32),axis=0)
            
        
        #predict the pre-processed input   
        output_arr =self.__model.run([self.__model.get_outputs()[0].name], {self.__model.get_inputs()[0].name: blob})[0]      
        

        boxes=[]
        scores=[]
        classes=[]
        centers=[]

        #filter out the bounding box of interest from the grid_w*grid_h possible number of bounding boxes
        #yolo layer of shape (grid_h,grid_w) has 5+num_classes channels (objectness,cx,cy,w,h,classes:)
                
        for i in range(self.grid_w):
            
            for j in range(self.grid_h):

                    confidence = output_arr[0,i, j, 0, 0].squeeze()

                    if confidence>=self.conf_threshold:

                        cx,cy,w,h=output_arr[0,i, j, 0,1:5]

                        #convert the box centers from scale 0-1 to 0-grid_w/grid_h
                        
                        
                        cx = ((j + (cx)) / self.grid_w)
                        cy = ((i + (cy)) / self.grid_h)

                        #convert the box centers from scale 0-grid_w/grid_h 0-inputimg_w/input_img_h

                        cx=cx*input_image.shape[1]
                        cy=cy*input_image.shape[0]
                        w=w*input_image.shape[1]
                        h=h*input_image.shape[0]

                        #(cx,cy,w,h) to (xmin,ymin,xmax,ymax)

                        xmin=max(0,int(cx-(w/2)))
                        ymin=max(0,int(cy-(h/2)))
                        xmax=max(0,int(cx+(w/2)))
                        ymax=max(0,int(cy+(h/2)))

                        if '<' in self.labels and ymax<0.6*input_image.shape[0]:
                            continue
                        
                        boxes.append([xmin,ymin,xmax,ymax])
                        scores.append(confidence)
                        classes.append(int(np.argmax(output_arr[0,i, j, 0, 5:])))
                        centers.append([cx,cy,w,h])
                        

        
        if len(boxes)<1:
            result['boxes']=boxes
            result['scores']=scores
            result['classes']=classes
            result['centers']=centers
            return result

                
                
                
        for box,score,class_ in zip(boxes,scores,classes):
            text=self.labels[class_]
            xmin,ymin,xmax,ymax=box
            

            cv2.rectangle(self.show_frame, (int(xmin),int(ymin)), (int(xmax),int(ymax)),self.__bbox_color, 2)
        

        result['boxes']=np.array(boxes)
        result['scores']=np.array(scores)
        result['classes']=np.array(classes)

        return result

    


if __name__=="__main__":

    from glob import glob

    files=sorted(glob('test/*.*g'))

    
    
    client=Detection(model_path='models/anglebracket_detector.onnx',
                                                classes_path='classes/ab_classes.txt',
                                                grid=(64,64))
    

    client.plot_boxes=True

    for file in files:

        client.detect(file)
        cv2.imshow('output',client.show_frame)
        k=cv2.waitKey(0)

        if k==ord('q'):
            cv2.destroyAllWindows()
            break