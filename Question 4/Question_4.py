import cv2
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':

    # Show the image using cv2 library and convert it to grayscale
    img = cv2.imread('Q_4.jpg', cv2.COLOR_BGR2GRAY)
    plt.imshow(img, cmap='gray')
    plt.title('Question 4 image')
    plt.show()

    # -------------------------------------------------- #

    # Compute the gradiant Gx, and Gy
    Gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    Gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

    # Compute the gradient magnitude and direction
    magnitude = cv2.magnitude(Gx, Gy)
    direction = cv2.phase(Gx, Gy)

    # Normalize the gradient magnitude to the range between 0 to 255
    StretchedMagnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

    # Convert to 8-bit image
    magnitude_8bits = np.uint8(StretchedMagnitude)

    # Display the image
    plt.imshow(magnitude_8bits, cmap='gray')
    plt.title('Image after normalize')
    plt.show()