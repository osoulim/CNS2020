from heapq import heappush, heappop
from .Population import Population


class Model:

    def __init__(self, layers_sizes, neuron_class, synapse_class):
        self.is_testing = False
        self.layers = list()
        self.synapses = {}
        self.spikes = []
        self.layers_sizes = layers_sizes
        self.layers_num = len(layers_sizes)
        for _id, x in enumerate(self.layers_sizes):
            tmp = Population(x, _id, neuron_class)
            self.layers.append(tmp)
        for i in range(self.layers_num - 1):
            for x in range(self.layers_sizes[i]):
                for y in range(self.layers_sizes[i + 1]):
                    self.synapses[(x, y)] = synapse_class()
                    self.synapses[(x, y)].add_connection(self.layers[i][x], self.layers[i+1][y])
        
    def add_spike_to_input(self, neuron_id, time):
        heappush(self.spikes, (time, 100, 0, neuron_id))
    
    def next_step(self):
        spike_time, exc, layer_id, neuron_id = heappop(self.spikes)
        spiked_neuron = self.layers[layer_id][neuron_id]
        spiked_neuron.update(spike_time)
        next_spikes = spiked_neuron.excite(exc)
        for spike in next_spikes:
            heappush(self.spikes, spike)

    def reset(self):
        for layer in self.layers:
            layer.reset()
        for synapse in self.synapses.values():
            synapse.reset()

    def disable_learning(self):
        self.is_testing = True
        for synapse in self.synapses.values():
            synapse.disable_learing()

    def is_queue_empty(self):
        return len(self.spikes) == 0