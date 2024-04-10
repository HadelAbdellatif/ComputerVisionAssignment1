import cv2
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':

    # Show image 1 using cv2 library
    img1 = cv2.imread('walk_1.jpg', cv2.IMREAD_GRAYSCALE)
    plt.imshow(img1, cmap='gray')
    plt.title('walk_1.jpg')
    plt.show()

    # Show image 2 using cv2 library
    img2 = cv2.imread('walk_2.jpg', cv2.IMREAD_GRAYSCALE)
    plt.imshow(img2, cmap='gray')
    plt.title('walk_2.jpg')
    plt.show()

    walk_1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    walk_2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)