import numpy as np


class MLP:
    def __init__(self, train_data, train_labels,
                 input_size,
                 output_size,
                 hidden_nodes=[5],
                 hidden_layers=1,
                 epochs=10,
                 learning_rate=0.1,
                 ):


        self.train_data = train_data
        self.train_labels = train_labels

        # TODO: Add columns of ones to train data for bias term

        self.input_size = input_size
        self.output_size = output_size

        self.epochs = epochs
        self.eta = learning_rate
        self.layers = hidden_layers

        self.weights_array = []

        # TODO: The code below is not correct but trying to create multiple
        # weight matrices for each layer to its next layer until output
        """
        current_rows = input_size
        current_cols = hidden_nodes[0]
        for layers in range(hidden_layers+1):
            for n in range(len(hidden_nodes)):
                self.weights_array[layers] = np.random.uniform(-0.05, 0.05,
                                                               (current_rows,
                                                                current_cols))
        """




    def train(self):
