import cv2
import numpy as np

def manual_bounding_box(image):
    print("Draw a bounding box using the mouse. Press 'c' to confirm and continue.")
    bbox = [0, 0, 0, 0]
    drawing = False
    ix, iy = -1, -1

    def draw_rectangle(event, x, y, flags, param):
        nonlocal ix, iy, drawing, bbox

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix, iy = x, y
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                temp_image = image.copy()
                cv2.rectangle(temp_image, (ix, iy), (x, y), (0, 255, 0), 2)
                cv2.imshow("Manual Bounding Box", temp_image)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            bbox = [min(ix, x), min(iy, y), abs(ix - x), abs(iy - y)]
            cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 0), 2)
            cv2.imshow("Manual Bounding Box", image)

    cv2.namedWindow("Manual Bounding Box")
    cv2.setMouseCallback("Manual Bounding Box", draw_rectangle)

    while True:
        cv2.imshow("Manual Bounding Box", image)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            break

    cv2.destroyWindow("Manual Bounding Box")
    return bbox
def automatic_bounding_box(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Use Otsu's thresholding instead of fixed threshold
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Apply morphological operations to clean up the binary image
    kernel = np.ones((5,5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Filter contours based on area to remove noise
        min_area = 100  # Adjust this value based on your image size
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
        
        if valid_contours:
            # Find the largest contour
            largest_contour = max(valid_contours, key=cv2.contourArea)
            
            # Get the bounding rectangle
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Add padding to the bounding box (optional)
            padding = 10
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(image.shape[1] - x, w + 2 * padding)
            h = min(image.shape[0] - y, h + 2 * padding)
            
            return (x, y, w, h)
    
    return None