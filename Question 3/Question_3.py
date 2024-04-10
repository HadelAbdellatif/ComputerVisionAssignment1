import cv2
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Working in Noisyimage1

    # Show the image using cv2 library
    img = cv2.imread('Noisyimage1.jpg', cv2.IMREAD_GRAYSCALE)
    plt.imshow(img, cmap='gray')
    plt.title('Noisyimage1')
    plt.show()


    noisy_image1 = cv2.imread('Noisyimage1.jpg', cv2.IMREAD_GRAYSCALE)

    # Apply a 5 by 5 mean filter to the noisy-image
    avg_filter = np.ones((5, 5), np.float32) / (5 ** 2)
    FilteredImage1 = cv2.filter2D(noisy_image1, -1, avg_filter)

    # Display the image after applied a 5 by 5 averaging filter
    plt.imshow(FilteredImage1, cmap='gray')
    plt.title('Noisyimage1 After Applied Averaging Filtered 5x5')
    plt.show()

    # Apply a 5x5 median filter to first image
    median_filtered_image1 = cv2.medianBlur(noisy_image1, 5)

    # Display the image after applied a 5 by 5 Median filter
    plt.imshow(median_filtered_image1, cmap='gray')
    plt.title('Noisyimage1 After Applied Median Filtered 5x5')
    plt.show()

    # ----------------------------------------------------- #

    # Working in Noisyimage2

    # Show the image using cv2 library
    img = cv2.imread('Noisyimage2.jpg', cv2.IMREAD_GRAYSCALE)
    plt.imshow(img, cmap='gray')
    plt.title('Noisyimage2')
    plt.show()

    noisy_image2 = cv2.imread('Noisyimage2.jpg', cv2.IMREAD_GRAYSCALE)

    # Apply a 5 by 5 mean filter to the noisy-image
    FilteredImage2 = cv2.filter2D(noisy_image2, -1, avg_filter)

    # Display the image after applied a 5 by 5 averaging filter
    plt.imshow(FilteredImage2, cmap='gray')
    plt.title('Noisyimage2 After Applied Averaging Filtered 5x5')
    plt.show()

    # Apply a 5x5 median filter to second image
    median_filtered_image2 = cv2.medianBlur(noisy_image2, 5)

    # Display the image after applied a 5 by 5 Median filter
    plt.imshow(median_filtered_image2, cmap='gray')
    plt.title('Noisyimage2 After Applied Median Filtered 5x5')
    plt.show()