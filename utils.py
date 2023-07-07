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

import numpy as np
import cv2
from skimage.transform import hough_line, hough_line_peaks
from skimage.feature import canny
from scipy.stats import mode


def return_lowest_ymin(arr):

    return min(arr[:,1])

def return_nearest_xmin(arr):

    return min(arr[:,0])

def return_farthest_xmax(arr):

    return max(arr[:,2])

def return_highest_ymax(arr):

    return max(arr[:,3])

def return_thinnest_area(arr):

    return min(arr[:,2]-arr[:,0])

def return_thickest_area(arr):

    return np.mean(arr[:,2]-arr[:,0])

def return_tallest_area(arr):

    return max(arr[:,2]-arr[:,0])

def check_multiple_line(arr):

    highest_ymin=max(arr[:,1])
    lowest_ymin=min(arr[:,1])

    if abs(lowest_ymin-highest_ymin)>return_tallest_area(arr):
        return True
    else:
        return False

def mask(img,arr):

    arr=np.clip(arr,a_min=0,a_max=None)

    bg=np.zeros((img.shape[0],img.shape[1]))

    for coords in arr:

        xmin,ymin,xmax,ymax=coords
        roi=img[ymin:ymax,xmin:xmax]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.bitwise_not(gray)
        # threshold the image, setting all foreground pixels to
        # 255 and all background pixels to 0
        thresh = cv2.threshold(gray, 0, 255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        #thresh = cv2.erode(thresh, None, iterations=1)

        bg[ymin:ymax,xmin:xmax]=thresh

    return bg

def recover_mrz(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(0.03*image.shape[1]),(int(0.05*image.shape[0]))))
    sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(0.1*image.shape[0]),int(0.1*image.shape[0])))

    # smooth the image using a 3x3 Gaussian, then apply the blackhat
    # morphological operator to find dark regions on a light background
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKernel)


    gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradX = np.absolute(gradX)
    (minVal, maxVal) = (np.min(gradX), np.max(gradX))
    gradX = (255 * ((gradX - minVal) / (maxVal - minVal))).astype("uint8")

    gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
    thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)
    # thresh = cv2.erode(thresh, None, iterations=4)

    #cv2.imshow('thresh',thresh)


    edges = cv2.Canny(image=thresh,threshold1=100, threshold2=250)
    
    image[edges>0]=[0,255,0]

    

    return image


def skew_angle_hough_transform(image):
    # convert to edges

    edges=cv2.Canny(image=image,threshold1=100, threshold2=200)
    #edges = canny(image)
    # Classic straight-line Hough transform between 0.1 - 180 degrees.
    tested_angles = np.deg2rad(np.arange(0.1, 180.0))
    h, theta, d = hough_line(edges, theta=tested_angles)
    
    # find line peaks and angles
    accum, angles, dists = hough_line_peaks(h, theta, d)
    
    # round the angles to 2 decimal places and find the most common angle.
    most_common_angle = mode(np.around(angles, decimals=2))[0]
    
    # convert the angle to degree for rotation.
    skew_angle = np.rad2deg(most_common_angle - np.pi/2)
    
    return skew_angle

def find_skew_angle(image):

    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    bg=np.zeros_like(image)

    (h, w) = image.shape[:2]

    bg=image.copy()

    angle=float(skew_angle_hough_transform(bg))

    return angle

def deskew(image,angle=None):

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h),
        flags=cv2.INTER_CUBIC)

    return rotated

def bbox_rotate(bbox, angle, rows, cols):
    """Rotates a bounding box by angle degrees.

    Args:
        bbox (tuple): A bounding box `(x_min, y_min, x_max, y_max)`.
        angle (int): Angle of rotation in degrees.
        rows (int): Image rows.
        cols (int): Image cols.
        interpolation (int): Interpolation method. TODO: Fix this, tt's not used in function

    Returns:
        A bounding box `(x_min, y_min, x_max, y_max)`.

    """
    x_min, y_min, x_max, y_max = bbox[:4]
    scale = cols / float(rows)
    x = np.array([x_min, x_max, x_max, x_min]) - 0.5
    y = np.array([y_min, y_min, y_max, y_max]) - 0.5
    angle = np.deg2rad(angle)
    x_t = (np.cos(angle) * x * scale + np.sin(angle) * y) / scale
    y_t = -np.sin(angle) * x * scale + np.cos(angle) * y
    x_t = x_t + 0.5
    y_t = y_t + 0.5

    x_min, x_max = min(x_t), max(x_t)
    y_min, y_max = min(y_t), max(y_t)

    return int(x_min), int(y_min), int(x_max), int(y_max)


# def rotate(point,image,angle):
#     (h, w) = image.shape[:2]
#     center = (w // 2, h // 2)

#     radians = np.deg2rad(angle)
#     x,y = point
#     offset_x, offset_y = center
#     adjusted_x = (x - offset_x)
#     adjusted_y = (y - offset_y)
#     cos_rad = np.cos(radians)
#     sin_rad = np.sin(radians)
#     qx = offset_x + cos_rad * adjusted_x + sin_rad * adjusted_y
#     qy = offset_y + -sin_rad * adjusted_x + cos_rad * adjusted_y
#     return int(qx), int(qy)