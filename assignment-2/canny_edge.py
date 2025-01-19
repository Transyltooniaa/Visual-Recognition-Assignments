import cv2
import numpy as np


def canny_edge_detection(image, bbox=None):
    
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Histogram equalization to improve contrast
    equalized = cv2.equalizeHist(blurred)
    
    # Fixed threshold values optimized for this type of image
    lower_thresh = 50  # Lower threshold for subtle edges
    upper_thresh = 260 # Upper threshold for strong edges
    
    if bbox:
        x, y, w, h = bbox
        roi = equalized[y:y+h, x:x+w]
        edges = cv2.Canny(roi, lower_thresh, upper_thresh)
        full_edges = np.zeros_like(equalized)
        full_edges[y:y+h, x:x+w] = edges
    else:
        full_edges = cv2.Canny(equalized, lower_thresh, upper_thresh)
    
    return full_edges

