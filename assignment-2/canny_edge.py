import cv2
import numpy as np


def canny_edge_detection(image, bbox=None):
     
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Grayscale Image", gray)
        
    # Fixed threshold values optimized for this type of image
    lower_thresh = 20  
    upper_thresh = 40  
    
    if bbox:
        x, y, w, h = bbox
        roi = gray[y:y+h, x:x+w]
        edges = cv2.Canny(roi, lower_thresh, upper_thresh)
        full_edges = np.zeros_like(gray)
        full_edges[y:y+h, x:x+w] = edges
    else:
        full_edges = cv2.Canny(gray, lower_thresh, upper_thresh)
    
    return full_edges

