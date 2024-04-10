import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np


def convolve2d(image, kernel):
    # The output image that will be returned
    output = np.zeros_like(image)

    # Add zero padding to the input image
    image_padded = np.zeros((image.shape[0] + 2, image.shape[1] + 2))
    image_padded[1:-1, 1:-1] = image

    # Loop over every pixel of the image
    for x in range(image.shape[1]):
        for y in range(image.shape[0]):
            # Perform convolution
            output[y, x] = (kernel * image_padded[y:y + 3, x:x + 3]).sum()

    return output


if __name__ == '__main__':

    # Display the image using cv2 library
    # Show the image using cv2 library
    img = cv2.imread('Question1image.jpg', cv2.IMREAD_GRAYSCALE)
    plt.imshow(img, cmap='gray')
    plt.title('Question 1 Image')
    plt.show()

    # ------------------------------------------------- #

    # Apply power law transformation with gamma=0.4
    gamma = 0.4
    corrected_image = np.array(255 * (img / 255) ** gamma, dtype='uint8')

    # Display the image after applied power law transformation with gamma=0.4
    plt.imshow(corrected_image, cmap='gray')
    plt.title('Image After Applied Power Law Transformed')
    plt.show()

    # ------------------------------------------------- #

    # Add zero-mean Gaussian noise with variance=40
    mean = 0 # according to zero-mean
    variance = 40
    sigma = variance ** 0.5
    gaussian_noise = np.random.normal(mean, sigma, img.shape)
    noisy_image = img + gaussian_noise

    # Display the image after added zero-mean Gaussian noise with variance=40
    plt.imshow(noisy_image, cmap='gray')
    plt.title('Image After Applied Zero Mean Gaussian Noise')
    plt.show()

    # ------------------------------------------------- #

    # Apply a 5 by 5 mean filter to the noisy-image
    mean_filter = np.ones((5, 5), np.float32) / (5 ** 2)
    FilteredImage = cv2.filter2D(noisy_image, -1, mean_filter)

    # Display the image after applied a 5 by 5 mean filter to the noisy-image
    plt.imshow(FilteredImage, cmap='gray')
    plt.title('Image After Applied Mean Filtered 5x5')
    plt.show()

    # ------------------------------------------------- #

    # Add salt and pepper noise (noise-density=0.1) to the original image
    salt_pepper_noise = np.copy(img)
    noise_density = 0.1
    threshold = 1 - noise_density
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            rand = np.random.rand()
            if rand < noise_density:
                salt_pepper_noise[i][j] = 0
            elif rand > threshold:
                salt_pepper_noise[i][j] = 255

    # Display the image after applied salt and pepper noise
    plt.imshow(salt_pepper_noise, cmap='gray')
    plt.title('Image After Added Salt and Pepper Noise')
    plt.show()

    # Apply a 7 by 7 median filter to the noisy-image
    MeadianFilter = cv2.medianBlur(salt_pepper_noise, 7)

    # Display the image after applied a 7 by 7 median filter to the noisy-image
    plt.imshow(MeadianFilter, cmap='gray')
    plt.title('Image After Applied 7x7 Median Filter')
    plt.show()

    # ------------------------------------------------- #

    # Apply a 7 by 7 mean filter to the salt and pepper noisy-image
    mean_filter = np.ones((7, 7), np.float32) / (7 ** 2)
    MeanfilteredImage = cv2.filter2D(salt_pepper_noise, -1, mean_filter)

    # Display the image after applied a 7 by 7 mean filter to the noisy-image
    plt.imshow(MeanfilteredImage, cmap='gray')
    plt.title('Mean Filter on Salt and Pepper Noise')
    plt.show()

    # ------------------------------------------------- #

    # Apply a Sobel filter to the original image and show the response

    # Sobel kernels
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    # Assuming 'image' is your NumPy array containing the image data
    image_sobel_x = convolve2d(img, sobel_x)
    image_sobel_y = convolve2d(img, sobel_y)

    # Calculate the magnitude of the gradients
    gradient_magnitude = np.sqrt(image_sobel_x ** 2 + image_sobel_y ** 2)
    gradient_magnitude *= 255.0 / gradient_magnitude.max()  # Normalize to 0-255

    # Display the result
    plt.imshow(gradient_magnitude, cmap='gray')
    plt.title('Sobel Filter Response')
    plt.show()
