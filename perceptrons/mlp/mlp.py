# Alex Shrestha
# Multilayer Perceptron Model
import numpy as np


class MLP:
    def __init__(self, train_data, train_labels,
#                 output_size,
                 nodes=[1, 1],
                 epochs=10,
                 learning_rate=0.1,
                 ):
        """
        Input size will be included in the nodes array where the first index
        contains the number of parameters in training data plus one (for the biases)
        Each subsequent index will hold the number of nodes for each hidden layer:
            nodes[0] = # of parameters
                .
                .
            nodes[i] = # of nodes in i_th hidden layer
                .
                .
            nodes[n] = # of nodes in output
        """

        """
        # Augmenting training data with biases of 1
        ones_column = np.ones((train_data.shape[0], 1)) 
        self.train_data = np.hstack((ones_column, train_data))

        self.train_labels = train_labels

        
        self.output_size = nodes[-1]

        self.epochs = epochs
        self.eta = learning_rate
        """

        self.layers = len(nodes) - 1

        for ele in nodes:
            print(f"value={ele}")

        # Each index in the array self.weights_array corresponds to weights
        # from one layer to its next layer
        self.weights_array = [np.random.uniform(0, 1, (2, 2)) for i in range(self.layers)]
        for layer in range(self.layers):
            print(f"Creating weight matrix of size {nodes[layer]+1, nodes[layer+1]}")
            self.weights_array[layer] = np.random.uniform(-0.05, 0.05,
                                                           (nodes[layer]+1, nodes[layer+1]))

            # Adding bias to each weight matrix
            if layer != 0 and (layer != self.layers-1):
                ones_column = np.ones((nodes[layer+1]+1, 1))
                self.weights_array[layer] = np.hstack((ones_column, 
                                                       self.weights_array[layer]))





    def train(self):
        # Will be using the sigmoid activation function for now
        pass
