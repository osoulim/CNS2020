import math
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def convolution2d(image, kernel):
    m, n = kernel.shape
    y, x = image.shape
    y = y - m + 1
    x = x - m + 1
    new_image = np.zeros((y, x))
    for i in range(y):
        for j in range(x):
            new_image[i][j] = np.sum(image[i:i+m, j:j+m]*kernel)
    return new_image


def get_dog_kernel(sigma1, sigma2, kernel_size):
    if kernel_size % 2 == 0:
        raise ValueError('kernel_size should be an odd number like 3, 5 and ...')
    result = np.zeros((kernel_size, kernel_size))
    for i in range(kernel_size):
        for j in range(kernel_size):
            x, y = i - kernel_size // 2, j - kernel_size // 2
            g1 = (1 / sigma1) * math.exp(-(x * x + y * y) / (2 * sigma1 * sigma1))
            g2 = (1 / sigma2) * math.exp(-(x * x + y * y) / (2 * sigma2 * sigma2))
            result[i][j] = (1 / (math.sqrt(2 * math.pi)) * (g1 - g2))
    return result


def get_gabor_kernel(landa, theta, sigma, gamma, kernel_size):
    if kernel_size % 2 == 0:
        raise ValueError('kernel_size should be an odd number like 3, 5 and ...')
    result = np.zeros((kernel_size, kernel_size))
    for i in range(kernel_size):
        for j in range(kernel_size):
            x, y = i - kernel_size // 2, j - kernel_size // 2
            X = x * math.cos(theta) + y * math.sin(theta)
            Y = - x * math.sin(theta) + y * math.cos(theta)
            result[i][j] = math.exp(-(X*X + gamma * gamma * Y * Y) / (2 * sigma * sigma)) * \
                math.cos(2 * math.pi * X / landa)
    return result  # - np.mean(result)


def show_images(image, kernels, filtered_images, parts=4):
    fig, axes = plt.subplots(len(kernels), 2 + parts, figsize=(10, 8))
    for i in range(len(kernels)):
        axes[i][0].imshow(kernels[i], "gray")
        axes[i][1].imshow(filtered_images[i], "gray")
        min_element, max_element = np.min(filtered_images[i]), np.max(filtered_images[i])
        for j in range(parts):
            min_thresh = min_element + (max_element - min_element) * j / parts
            max_thresh = min_element + (max_element - min_element) * (j + 1) / parts
            axes[i][2 + j].imshow(
                (min_thresh <= filtered_images[i]) * (filtered_images[i] < max_thresh) * filtered_images[i], "gray")
    plt.show()


def show_images_gabor(image, kernels, filtered_images):
    fig, axes = plt.subplots(len(kernels) // 4, 4, figsize=(10, 8))
    fig1, axes1 = plt.subplots(len(kernels) // 4, 4, figsize=(10, 8))
    for i in range(0, len(kernels), 4):
        row = (i // 4)
        for j in range(4):
            axes[row][j].imshow(kernels[i + j], "gray")
            axes1[row][j].imshow(filtered_images[i + j], "gray")
    plt.show()

# if __name__ == "__main__":
#     gray = np.asarray(Image.open("data/gorill.png").convert('L'))
#     dog_kernels = [get_dog_kernel(size / 3.2, size / 2, size) for size in [5, 13, 21, 29]]
#     outputs = [convolution2d(gray, kernel) for kernel in dog_kernels]
#     show_images(gray, dog_kernels, outputs)


if __name__ == "__main__":
    gray = np.asarray(Image.open("data/gorill.png").convert('L'))
    gabor_kernels = [get_gabor_kernel(10, math.pi * theta / 4, 5, 0.5, size)
                     for size in [7, 13, 19, 25, 31, 37, 43, 49]
                     for theta in range(4)]
    outputs = [convolution2d(gray, kernel) for kernel in gabor_kernels]
    show_images_gabor(gray, gabor_kernels, outputs)
