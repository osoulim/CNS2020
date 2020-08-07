import math

from bindsnet.learning import PostPre
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.network.topology import Conv2dConnection, MaxPool2dConnection

from .util import get_gabor_kernel, convolution2d

SUBJECTS = ['airplane', 'butterfly', 'Faces_easy', 'garfield']

GAMMA = 0.5
THETA_PARTS = 4
THETA = [math.pi * i / THETA_PARTS for i in range(THETA_PARTS)]
GABOR_SIZES = [5, 7, 9, 11]
IMAGE_SIZE = 50
KERNELS = [get_gabor_kernel(size / 2, theta, size / 3, GAMMA, size) for theta in THETA for size in GABOR_SIZES]
FILTERS = [lambda x: convolution2d(x, kernel) for kernel in KERNELS]

FEATURES = 8


def get_s1_name(size): return 'S1_%d' % size


def get_c1_name(size): return 'C1_%d' % size


def get_s2_name(size, index): return 'S2_%d_%d' % (size, index)


def get_c2_name(size, index): return 'C2_%d_%d' % (size, index)


def create_hmax(network):
    for size in GABOR_SIZES:
        s1 = Input(shape=(THETA_PARTS, IMAGE_SIZE, IMAGE_SIZE), traces=True)
        network.add_layer(layer=s1, name=get_s1_name(size))

        c1 = LIFNodes(shape=(THETA_PARTS, IMAGE_SIZE // 2, IMAGE_SIZE // 2), thresh=-64, traces=True)
        network.add_layer(layer=c1, name=get_c1_name(size))

        max_pool = MaxPool2dConnection(s1, c1, kernel_size=2, stride=2, decay=0)
        network.add_connection(max_pool, get_s1_name(size), get_c1_name(size))

    for index in range(FEATURES):
        for size in GABOR_SIZES:
            s2 = LIFNodes(shape=(1, IMAGE_SIZE // 2, IMAGE_SIZE // 2), traces=True)
            network.add_layer(layer=s2, name=get_c2_name(size, index))

            conv = Conv2dConnection(network.layers[get_c1_name(size)], s2, 5, padding=2, weight_decay=0.01,
                                    nu=0.01, update_rule=PostPre, decay=0.5)
            network.add_connection(conv, get_c1_name(size), get_s2_name(size, index))

            c2 = LIFNodes(shape=(1, IMAGE_SIZE // 4, IMAGE_SIZE // 4), thresh=-64, traces=True)
            network.add_layer(layer=c2, name=get_c2_name(size, index))

            max_pool = MaxPool2dConnection(s2, c2, kernel_size=2, stride=2, decay=0)
            network.add_connection(max_pool, get_s2_name(size, index), get_c2_name(size, index))

if __name__ == "__main__":
    pass
