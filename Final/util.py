import math
import numpy as np


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
