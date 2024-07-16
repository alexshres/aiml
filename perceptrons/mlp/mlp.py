# Alex Shrestha
# Multilayer Perceptron Model
import numpy as np


class MLP:
    def __init__(self, train_data, train_labels,
                 nodes=[1, 1],
                 epochs=10,
                 learning_rate=0.1,
                 momentum=0.9
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
        self.epochs = epochs
        self.eta = learning_rate
        self.nodes = nodes
        self.layers = len(nodes) - 1
        self.momentum = momentum

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
        self.__activations = [[0 for i in range(self.nodes[x])] for x in range(1, len(self.nodes))]
        self.__errors = [[0 for i in range(nodes[x])] for x in range(self.layers)]

        # Adding ones to each activated value since that will become the bias term
        for i in range(self.layers-1):
            self.__activations[i].insert(-1, 1)

        
    def __sigmoid(self, x):
        """ Implementation of sigmoid function"""
        return 1/(1+np.exp(-x))


    def __forward(self, datum, weight):
        """ 
        Forward propagation of finding h_j where h_j is the activated value
        of the j_th node
        """
        return self.__sigmoid(np.dot(datum.T, weight))

    def __weight_update(self):
        """
        Backpropagation portion of the algorithm
        """
        pass


    def train(self):
        for row in range(self.train_data.shape[0]):
            target = self.train_labels[row]
            encoding = [0.9 if x == target else 0.1 for x in range(self.nodes[-1])]
            data = self.train_data[row]

            # Work through each layer activating the nodes
            for lr in range(self.layers): 
                for i in range(self.nodes[lr+1]):
                    self.__activations[lr][i] =         \
                             self.__forward(
                                            data, 
                                            self.weights_array[lr][:, i]
                                           )
                data = np.asarray(self.__activations[lr])
            # At this point all output nodes have some value
            # Now we update weights based on error starting with the last layer before
            # the output layer

            print(f"Target encoding {encoding}")
            print(f"Output nodes    {self.__activations[-1]}")

            for i in range(self.nodes[-1]):
                self.__errors[-1][i] = self.__activations[-1][i] * (1-self.__activations[-1][i]) * \
                                (encoding[i] - self.__activations[-1][i])

            print(f"Error array {self.__errors}")

        pass




