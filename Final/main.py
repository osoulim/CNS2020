import math

import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from bindsnet.analysis.plotting import plot_spikes
from bindsnet.encoding import RankOrderEncoder
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.network import Network
from bindsnet.network.monitors import NetworkMonitor
from bindsnet.network.monitors import Monitor
from bindsnet.network.topology import Conv2dConnection, MaxPool2dConnection, Connection
from bindsnet.learning import PostPre
from bindsnet.evaluation import assign_labels, all_activity
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

from dataset import Dataset

SUBJECTS = ['airplanes', 'butterfly', 'Faces_easy', 'garfield']
# SUBJECTS = ['butterfly']

GAMMA = 0.5
THETA_PARTS = 4
THETA = [math.pi * i / THETA_PARTS for i in range(THETA_PARTS)]
GABOR_SIZES = [5, 7, 9, 11]
IMAGE_SIZE = 50
KERNELS = [cv2.getGaborKernel((size, size), size / 3, theta, size / 2, GAMMA)
           for theta in THETA for size in GABOR_SIZES]
FILTERS = [lambda x: cv2.filter2D(x, -1, kernel) for kernel in KERNELS]

FEATURES = 8
RUN_TIME = 50


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

        max_pool = MaxPool2dConnection(s1, c1, kernel_size=2, stride=2, decay=0.0)
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


def encode_image(image):
    t = torch.from_numpy(image).float()
    if t.min() < 0:
        t -= t.min(t)
    encoder = RankOrderEncoder(RUN_TIME)
    return encoder(t)


def encode_image_batch(image_batch):
    network_input = {}
    for i, size in enumerate(GABOR_SIZES):
        inputs = torch.empty((RUN_TIME, 1, THETA_PARTS, IMAGE_SIZE, IMAGE_SIZE))
        for j in range(THETA_PARTS):
            inputs[:, 0, j, :, :] = encode_image(image_batch[i * THETA_PARTS + j])
        network_input[get_s1_name(size)] = inputs
    return network_input


def train(network, data):
    for image_batch in tqdm(data):
        network_input = encode_image_batch(image_batch)
        network.run(network_input, time=RUN_TIME)


if __name__ == "__main__":
    print("Creating network")
    network = Network(batch_size=1)
    create_hmax(network)

    print("Loading data")
    dataset = Dataset('data', subjects=SUBJECTS, image_size=(IMAGE_SIZE, IMAGE_SIZE))
    train_data, train_labels, test_data, test_labels = dataset.get_data(filters=FILTERS)

    print("Training %d samples" % len(train_data))
    train(network, train_data)