import cv2
from bounding_box import manual_bounding_box, automatic_bounding_box
from canny_edge import canny_edge_detection
from kmeans_segmentation import k_means_segmentation

if __name__ == "__main__":
    # Load the image
    image_path = "InputUngradedAssignment1.jpg"  # Replace with your image file
    image = cv2.imread(image_path)

    if image is None:
        print("Error: Image not found!")
        exit()
    
    
    # Ask user for manual or automatic detection
    print("Choose an option for bounding box detection:")
    print("1. Manual Detection")
    print("2. Automatic Detection")
    choice = input("Enter your choice (1 or 2): ")
    
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Original Image", grayscale)

    if choice == "1":
        bbox = manual_bounding_box(image)
        edges = canny_edge_detection(image, bbox)
        cv2.imshow("Canny Edge Detection", edges)
    elif choice == "2":
        bbox = automatic_bounding_box(image)
        if bbox:
            edges = canny_edge_detection(image, bbox)
            cv2.imshow("Canny Edge Detection", edges)

        else:
            print("No object detected for automatic bounding box.")
    else:
        print("Invalid choice. Exiting.")
        exit()

    # K-means Segmentation
    segmented_image = k_means_segmentation(image, k=3)
    cv2.imshow("K-means Segmentation", segmented_image)

    # Wait for a key press and close all windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()
