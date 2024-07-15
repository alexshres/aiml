# Alex Shrestha
# Multilayer Perceptron Model
import numpy as np


class MLP:
    def __init__(self, train_data, train_labels,
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

        # Augmenting training data with biases of 1
        ones_column = np.ones((train_data.shape[0], 1)) 
        self.train_data = np.hstack((ones_column, train_data))

        self.train_labels = train_labels

        
        self.output_size = nodes[-1]

        self.epochs = epochs
        self.eta = learning_rate

        self.nodes = nodes
        self.layers = len(nodes) - 1


        # Each index in the array self.weights_array corresponds to weights
        # from one layer to its next layer
        self.weights_array = [np.random.uniform(0, 1, (2, 2)) for i in range(self.layers)]
        for layer in range(self.layers):
            self.weights_array[layer] = np.random.uniform(-0.05, 0.05,
                                                           (nodes[layer]+1, nodes[layer+1]))

            # Adding bias to each weight matrix
            if layer != 0 and (layer != self.layers-1):
                ones_column = np.ones((nodes[layer+1]+1, 1))
                self.weights_array[layer] = np.hstack((ones_column, 
                                                       self.weights_array[layer]))

        # Creating an activation ragged array where each index represents a layer and 
        # contains node values post activation layer
        self.__activations = [[0 for i in range(nodes[x])] for x in range(1, len(nodes))]

        
    def __sigmoid(self, x):
        """ Implementation of sigmoid function"""
        return 1/(1+np.exp(-x))


    def __forward(self, datum, weight):
        """ 
        Forward propagation of finding h_j where h_j is the activated value
        of the j_th node
        """
        return self.__sigmoid(np.dot(datum.T, weight))


    def train(self):
        for row in range(self.train_data.shape[0]):
            current_layer = 0
            for i in range(self.nodes[1]):
                self.__activations[current_layer][i] = self.__forward(
                                                            self.train_data[row], 
                                                            self.weights_array[current_layer][:, i]
                                                            )
                print(f"Activated value for layer {current_layer+1} and node {i+1} is"  \
                      f" {self.__activations[current_layer][i]}")
            
        pass




