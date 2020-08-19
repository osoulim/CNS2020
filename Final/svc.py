import math
import glob
import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from bindsnet.analysis.plotting import plot_spikes, plot_conv2d_weights, plot_voltages, plot_assignments
from bindsnet.encoding import RankOrderEncoder
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.network import Network, load
from bindsnet.network.monitors import Monitor
from bindsnet.network.topology import Conv2dConnection, MaxPool2dConnection, Connection
from bindsnet.learning import PostPre, WeightDependentPostPre
from bindsnet.evaluation import assign_labels, all_activity
from sklearn.metrics import classification_report

from dataset import Dataset

SUBJECTS = ['soccer_ball', 'butterfly', 'revolver', 'garfield']
# SUBJECTS = ['soccer_ball', 'revolver']

GAMMA = 0.5
FILTER_TYPES = 4
THETA = [math.pi * i / FILTER_TYPES for i in range(FILTER_TYPES)]
FILTER_SIZES = [5, 11, 19]
IMAGE_SIZE = 35
KERNELS = [cv2.getGaborKernel((size, size), size / 3, theta, size / 2, GAMMA)
           for theta in THETA for size in FILTER_SIZES]
# KERNELS = [cv2.getGaussianKernel(size, size / 9) - cv2.getGaussianKernel(size, size / 4.5)
#            for size in FILTER_SIZES]
FILTERS = [lambda x: cv2.filter2D(x, -1, kernel) for kernel in KERNELS]

FEATURES = range(12)
RUN_TIME = 40
TRAINED_NETWORK_PATH = 'trained_network_svc.pt'
EPOCHS = 2


def get_s1_name(size): return 'S1_%d' % size


def get_c1_name(size): return 'C1_%d' % size


def get_s2_name(size, feature): return 'S2_%d_%d' % (size, feature)


def get_c2_name(size, feature): return 'C2_%d_%d' % (size, feature)


def create_hmax(network):
    for size in FILTER_SIZES:
        s1 = Input(shape=(FILTER_TYPES, IMAGE_SIZE, IMAGE_SIZE), traces=True)
        network.add_layer(layer=s1, name=get_s1_name(size))
        # network.add_monitor(Monitor(s1, ["s"]), get_s1_name(size))

        c1 = LIFNodes(shape=(FILTER_TYPES, IMAGE_SIZE // 2, IMAGE_SIZE // 2), thresh=-64, traces=True)
        network.add_layer(layer=c1, name=get_c1_name(size))
        # network.add_monitor(Monitor(c1, ["s", "v"]), get_c1_name(size))

        max_pool = MaxPool2dConnection(s1, c1, kernel_size=2, stride=2, decay=0.2)
        network.add_connection(max_pool, get_s1_name(size), get_c1_name(size))

    for feature in FEATURES:
        for size in FILTER_SIZES:
            s2 = LIFNodes(shape=(1, IMAGE_SIZE // 2, IMAGE_SIZE // 2), thresh=-64, traces=True)
            network.add_layer(layer=s2, name=get_s2_name(size, feature))
            # network.add_monitor(Monitor(s2, ["s", "v"]), get_s2_name(size, feature))

            conv = Conv2dConnection(network.layers[get_c1_name(size)], s2, 15, padding=7,
                                    update_rule=PostPre, wmin=0, wmax=1)

            network.add_monitor(
                Monitor(conv, ["w"]),
                "conv%d%d" % (feature, size)
            )

            network.add_connection(conv, get_c1_name(size), get_s2_name(size, feature))

            c2 = LIFNodes(shape=(1, 1, 1), thresh=-64, traces=True)
            network.add_layer(layer=c2, name=get_c2_name(size, feature))
            # network.add_monitor(Monitor(c2, ["s", "v"]), get_c2_name(size, feature))

            max_pool = MaxPool2dConnection(s2, c2, kernel_size=IMAGE_SIZE // 2, decay=0.0)
            network.add_connection(max_pool, get_s2_name(size, feature), get_c2_name(size, feature))


def encode_image(image):
    t = torch.from_numpy(image).float()
    if t.min() < 0:
        t -= t.min(t)
    encoder = RankOrderEncoder(RUN_TIME)
    return encoder(t)


def encode_image_batch(image_batch):
    network_input = {}
    for i, size in enumerate(FILTER_SIZES):
        inputs = torch.empty((RUN_TIME, 1, FILTER_TYPES, IMAGE_SIZE, IMAGE_SIZE))
        for j in range(FILTER_TYPES):
            inputs[:, 0, j, :, :] = encode_image(image_batch[i * FILTER_TYPES + j])
        network_input[get_s1_name(size)] = inputs
    return network_input


def train(network, data):
    for image_batch in tqdm(data):
        network_input = encode_image_batch(image_batch)
        network.run(network_input, time=RUN_TIME)


def test(network, data, labels):
    activities = torch.zeros(len(data), RUN_TIME, len(SUBJECTS)) # data_size * run_time * classes
    true_labels = torch.from_numpy(np.array(labels))

    for index, image_batch in enumerate(tqdm(data)):
        network_input = encode_image_batch(image_batch)
        network.run(network_input, time=RUN_TIME)
        spikes = network.monitors["OUT"].get("s")
        activities[index, :, :] = spikes[-RUN_TIME:, 0]

    assignments = assign_labels(activities, true_labels, len(SUBJECTS))
    predicated_labels = all_activity(activities, assignments[0], len(SUBJECTS))
    print(classification_report(true_labels, predicated_labels))


if __name__ == "__main__":
    print("Loading data")
    dataset = Dataset('data', subjects=SUBJECTS, image_size=(IMAGE_SIZE, IMAGE_SIZE))
    train_data, train_labels, test_data, test_labels = dataset.get_data(filters=FILTERS)

    if not glob.glob(TRAINED_NETWORK_PATH):
        network = Network()
        create_hmax(network)

        for e in range(EPOCHS):
            train(network, train_data)

        network.save(TRAINED_NETWORK_PATH)
    else:
        network = load(TRAINED_NETWORK_PATH)
        print("Trained network loaded from file")

    for feature in FEATURES:
        for size in FILTER_SIZES:
            weights = network.monitors["conv%d%d" % (feature, size)].get("w")
            plot_conv2d_weights(weights[0], cmap='Greys')
    plt.show()
    #
    # for feature in FEATURES:
    #     for size in FILTER_SIZES:
    #         # voltages = network.monitors[get_s2_name(size, feature)].get("v")
    #         # spikes = network.monitors[get_s2_name(size, feature)].get("s")
    #         plot_voltages({"C2": voltages[-300: ]})
    #         plot_spikes({"C2": spikes[-300: ]})


    voltages = network.monitors["OUT"].get("v")
    spikes = network.monitors["OUT"].get("s")
    plot_voltages({"Output": voltages})
    plot_spikes({"output": spikes})

    plt.show()

    network.train(False)
    print("Start testing")
    test(network, test_data, test_labels)
