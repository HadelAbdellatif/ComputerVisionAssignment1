import cv2
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':

    # Show the image using cv2 library and convert it to grayscale
    img = cv2.imread('Q_4.jpg')
    GrayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plt.imshow(GrayImg, cmap='gray')
    plt.title('Question 6 Gray image')
    plt.show()

    # Test different values of ‘Threshold’
    thresholds = [(50, 150), (100, 200), (150, 250), (200, 300)]

    # Apply Canny edge detector for Thresholds
    for i, (low_thresh, high_thresh) in enumerate(thresholds):
        edges = cv2.Canny(GrayImg, low_thresh, high_thresh)

        # Display the image
        plt.figure(figsize=(8, 4))
        plt.imshow(edges, cmap='gray')
        plt.title(f'Canny Edges with Thresholds {low_thresh} and {high_thresh}')
        plt.axis('off')
        plt.show()

