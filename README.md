### MRZ READER ##

This is a YOLO object detection inference repository to extract & read MRZ from passports. The training notebook is added in train dir but this repository focuses on inference. 
I did this prototype as a part of assessment where the challenge was to recover MRZ are from passport
by detecting angled brackets (<)

```
pip3 install -r requirements.txt
python3 main.py

```
#### MRZ EXTRACTION & OCR ####



<p>
    <img src="final_output.png"/>
</p>


<p>

#### TECHNIQUES USED IN THE PROJECT ####

1) Custom YOLO Tensorflow for OBJECT DETECTION for detecting documents & "<" .  
2) Tensorflow CONVOLUTIONAL-RNN for OCR
3) OTSU Thresholding to binarize & segment MRZ "<"
4) Hough Lines to detect skew angle 
</p>



<p>

#### SOLUTION ####

The solution involves both classic Computer Vision approach & Deep Learning approach. 

Orderwise as in image below, the pipeline starts with document detection by YOLO model.
Followed by deskewing image using hough transform.

From the deskewed image, the angled brackets (<) are detected from which the MRZ area 
is recovered by computing the distance & location of angled brackets & MRZ standards.

The recovered MRZ area is passed to OTSU thresholding & upper half / lower half of MRZ 
is cropped & sent to Convolutional OCR model
</p>

<p>
    <img src="output.jpg"/>
</p>


<p>

#### MODEL INFORMATION & LATENCY ####

The Pytorch & Tensorflow models were converted to ONNX for better model loading time &
cross-platform conversion

Document Detector     - 1.6MB

Anglebracket Detector - 2.5MB

OCR model             - 9MB

The entire process of 3 models has combined latency of less than 200ms/document on i7 CPU.

The models can further be converted to formats like CoreML, TensorRT, OpenVINO etc based
on deployment strategy. Also can be quantised to fp16 to reduce model size by half with
negligible drop in accuracy.

</p>
