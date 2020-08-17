import math
import glob
import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from bindsnet.analysis.plotting import plot_spikes, plot_conv2d_weights
from bindsnet.encoding import RankOrderEncoder
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.network import Network, load
from bindsnet.network.monitors import Monitor
from bindsnet.network.topology import Conv2dConnection, MaxPool2dConnection, Connection
from bindsnet.learning import PostPre, WeightDependentPostPre
from bindsnet.evaluation import assign_labels, all_activity
from sklearn.metrics import classification_report

from dataset import Dataset

# SUBJECTS = ['airplanes', 'butterfly', 'Faces_easy', 'garfield']
SUBJECTS = ['butterfly', 'garfield']

GAMMA = 0.5
FILTER_TYPES = 1
THETA = [math.pi * i / FILTER_TYPES for i in range(FILTER_TYPES)]
FILTER_SIZES = [5, 7, 9, 11]
IMAGE_SIZE = 30
# KERNELS = [cv2.getGaborKernel((size, size), size / 3, theta, size / 2, GAMMA)
#            for theta in THETA for size in FILTER_SIZES]
KERNELS = [cv2.getGaussianKernel(size, size/3.2) - cv2.getGaussianKernel(size, size/2)
           for size in FILTER_SIZES]
FILTERS = [lambda x: cv2.filter2D(x, -1, kernel) for kernel in KERNELS]

RUN_TIME = 50
DECISION_LAYER_SIZE = 1000

TRAINED_NETWORK_PATH = 'trained_network.pt'


def get_s1_name(size): return 'S1_%d' % size


def get_c1_name(size): return 'C1_%d' % size


def get_s2_name(size): return 'S2_%d' % size


def get_c2_name(size): return 'C2_%d' % size


def create_hmax(network):
    for size in FILTER_SIZES:
        s1 = Input(shape=(FILTER_TYPES, IMAGE_SIZE, IMAGE_SIZE), traces=True)
        network.add_layer(layer=s1, name=get_s1_name(size))

        c1 = LIFNodes(shape=(FILTER_TYPES, IMAGE_SIZE // 2, IMAGE_SIZE // 2), thresh=-63, traces=True)
        network.add_layer(layer=c1, name=get_c1_name(size))

        max_pool = MaxPool2dConnection(s1, c1, kernel_size=2, stride=2, decay=0.5)
        network.add_connection(max_pool, get_s1_name(size), get_c1_name(size))

        s2 = LIFNodes(shape=(1, IMAGE_SIZE // 2, IMAGE_SIZE // 2), traces=True)
        network.add_layer(layer=s2, name=get_s2_name(size))

        conv = Conv2dConnection(network.layers[get_c1_name(size)], s2, 5, padding=2, weight_decay=0.01,
                                nu=[0.0001, 0.0002], update_rule=WeightDependentPostPre, wmin=0, wmax=1)
        network.add_connection(conv, get_c1_name(size), get_s2_name(size))

        network.add_monitor(Monitor(conv, ["w"]), get_s2_name(size))

        c2 = LIFNodes(shape=(1, IMAGE_SIZE // 4, IMAGE_SIZE // 4), thresh=-63.5, traces=True)
        network.add_layer(layer=c2, name=get_c2_name(size))

        max_pool = MaxPool2dConnection(s2, c2, kernel_size=2, stride=2, decay=0.0)
        network.add_connection(max_pool, get_s2_name(size), get_c2_name(size))


def add_decision_layers(network):
    d1 = LIFNodes(n=DECISION_LAYER_SIZE, traces=True)
    network.add_layer(d1, "D1")

    for size in FILTER_SIZES:
        connection = Connection(
            source=network.layers[get_c2_name(size)],
            target=d1,
            w=0.05 + 0.1 * torch.randn(network.layers[get_c2_name(size)].n, d1.n),
            update_rule=PostPre
        )
        network.add_connection(connection, get_c2_name(size), "D1")

    output = LIFNodes(n=len(SUBJECTS), traces=True)
    network.add_layer(output, "OUT")

    connection = Connection(
        source=d1,
        target=output,
        w=0.05 + 0.1 * torch.randn(d1.n, output.n),
        update_rule = PostPre
    )
    network.add_connection(connection, "D1", "OUT")

    rec_connection = Connection(
        source=output,
        target=output,
        w=0.05 * (torch.eye(output.n) - 1),
        decay=0.0,
    )
    network.add_connection(rec_connection, "OUT", "OUT")


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
    activities = torch.zeros(len(data), RUN_TIME, len(SUBJECTS))
    true_labels = torch.from_numpy(np.array(labels))

    for index, image_batch in enumerate(tqdm(data)):
        network_input = encode_image_batch(image_batch)
        network.run(network_input, time=RUN_TIME)
        spikes = network.monitors["Result"].get("s")
        activities[index, :, :] = spikes[-RUN_TIME, 0]

    assignments = assign_labels(activities, true_labels, len(SUBJECTS))
    predicated_labels = all_activity(activities, assignments[0], len(SUBJECTS))
    print(classification_report(true_labels, predicated_labels))


def plot_features(network):
    for size in FILTER_SIZES:
        monitor = network.monitors[get_s2_name(size)]
        weights = monitor.get("w")
        plt.ioff()
        plot_conv2d_weights(weights[0])
        plt.show()


if __name__ == "__main__":
    print("Loading data")
    dataset = Dataset('data', subjects=SUBJECTS, image_size=(IMAGE_SIZE, IMAGE_SIZE))
    train_data, train_labels, test_data, test_labels = dataset.get_data(filters=FILTERS)

    if not glob.glob(TRAINED_NETWORK_PATH):
        network = Network()
        create_hmax(network)

        print("Training %d samples" % len(train_data))
        train(network, train_data)

        print("Add decision layers")
        add_decision_layers(network)

        print("Training again...")
        train(network, train_data)

        network.save(TRAINED_NETWORK_PATH)
    else:
        print("Trained network loaded from file")
        network = load(TRAINED_NETWORK_PATH)

    network.add_monitor(
        Monitor(network.layers["OUT"], ["s"]),
        "Result"
    )

    for size in FILTER_SIZES:
        network.add_monitor(
            Monitor(network.layers[get_c2_name(size)], ["s"]),
            get_c2_name(size)
        )


    network.training = False
    print("Start testing")
    test(network, test_data, test_labels)
