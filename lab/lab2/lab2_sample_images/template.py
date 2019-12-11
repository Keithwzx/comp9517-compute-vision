# Template for lab02 task 3

import cv2
import math
import numpy as np
import sys

class SiftDetector():
    def __init__(self, norm="L2", params=None):
        self.detector=self.get_detector(params)
        self.norm=norm

    def get_detector(self, params):
        if params is None:
            params={}
            params["n_features"]=0
            params["n_octave_layers"]=3
            params["contrast_threshold"]=0.04
            params["edge_threshold"]=10
            params["sigma"]=1.6

        detector = cv2.xfeatures2d.SIFT_create(
                nfeatures=params["n_features"],
                nOctaveLayers=params["n_octave_layers"],
                contrastThreshold=params["contrast_threshold"],
                edgeThreshold=params["edge_threshold"],
                sigma=params["sigma"])

        return detector

# Rotate an image
#
# image: image to rotate
# x:     x-coordinate of point we wish to rotate around
# y:     y-coordinate of point we wish to rotate around
# angle: degrees to rotate image by
#
# Returns a rotated copy of the original image
def rotate(image, x, y, angle):
    R=cv2.getRotationMatrix2D((y,x),angle,1)
    rows,cols,channels=image.shape
    output=cv2.warpAffine(img0,R,(cols,rows))
    return output

# Get coordinates of center point.
#
# image:  Image that will be rotated
# return: (x, y) coordinates of point at center of image
def get_img_center(image):
    rows,cols,channels = image.shape
    x = rows/2
    y = cols/2
    return x,y


if __name__ == '__main__':
    # Read image with OpenCV and convert to grayscale
    img0=cv2.imread('Eiffel_Tower.jpg')
    gray=cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)


    # Initialize SIFT detector
    sift=SiftDetector()
    sift_detector=sift.detector
    


    # Store SIFT keypoints of original image in a Numpy array
    kp=sift_detector.detect(gray, None)
    img_key=cv2.drawKeypoints(gray,kp,img0)
#    cv2.imwrite('task1_1/4_features.png',img)
        # caculate keypoints:
    kp,des=sift_detector.compute(gray,kp)
    kp_array=np.array(kp)
    print(kp_array.shape)

    # Rotate around point at center of image.
    coordinates=get_img_center(img0)
    x=coordinates[0]
    y=coordinates[1]
    output=rotate(img0,x,y,-60)
    cv2.imwrite('task2_Rotate_60_degrees.png',output)
    


    # Degrees with which to rotate image
    degrees=-60


    # Number of times we wish to rotate the image
    

    
    # BFMatcher with default params
    

        # Rotate image
        

        # Compute SIFT features for rotated image
        

        
        # Apply ratio test
        


        # cv2.drawMatchesKnn expects list of lists as matches.

    cv2.imshow('image',output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
