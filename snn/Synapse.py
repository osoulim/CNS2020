import numpy as np
import math
from snn import parameters


class Synapse:

    def __init__(self, pre_neurons=None, post_neurons=None):
        super().__init__()
        self.observe = None
        if post_neurons is None:
            post_neurons = list()
        if pre_neurons is None:
            pre_neurons = list()

        learning_scale = 2 / parameters.nDots

        self.weight = np.random.normal(0.95, 0.05)
        self.delay = np.random.normal(25, 0.02)
        self.number_of_spikes = 0
        self.last_update_time = -1
        self.cnt = 0
        self.is_frozen = False
        self.pre_neurons = pre_neurons
        self.post_neurons = post_neurons
        # self.feature = feature

        # STDP parameters
        self.tau_pre = 5.0
        self.tau_post = 5.0
        self.cApre = 4e-4 * learning_scale
        self.cApost = 5e-4 * learning_scale
        self.wmax = 1
        self.wmin = 0
        self.eps_weight_decay = 1e-4 * learning_scale

    def disable_learning(self):
        self.is_frozen = True

    def add_connection(self, pre_neuron, post_neuron):
        self.pre_neurons.append(pre_neuron)
        self.post_neurons.append(post_neuron)
        pre_neuron.forward_neurons.append(post_neuron)
        post_neuron.back_neurons.append(pre_neuron)
        pre_neuron.forward_synapses.append(self)
        post_neuron.back_synapses.append(self)

    def STDP(self, pre_neuron, post_neuron):
        if self.is_frozen:
            return 0

        if pre_neuron.spike_time == -1:
            self.weight = max(self.wmin, self.weight - self.eps_weight_decay)
            return 0
        delta_time = post_neuron.get_spike_time() - pre_neuron.get_spike_time() - self.delay + 1e-9
        if delta_time >= 0:
            self.weight = np.clip(self.weight + (self.cApost * math.exp(-delta_time / self.tau_post)), self.wmin,
                                  self.wmax)
        else:
            self.weight = np.clip(self.weight - (self.cApre * math.exp(delta_time / self.tau_pre)), self.wmin,
                                  self.wmax)
