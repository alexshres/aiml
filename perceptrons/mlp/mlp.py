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
        self.outputs = nodes[-1]
        self.epochs = epochs
        self.eta = learning_rate
        self.nodes = nodes
        self.layers = len(nodes) - 1
        self.momentum = momentum

        # Each index in the array self.weights corresponds to weights
        # from one layer to its next layer
        self.weights = [np.random.uniform(0, 1, (2, 2)) for i in range(self.layers)]
        for layer in range(self.layers):
            self.weights[layer] = np.random.uniform(-0.05, 0.05,
                                                           (nodes[layer]+1, nodes[layer+1]))

        # Creating an activation ragged array where each index represents a layer and 
        # contains node values post activation layer
        """
        self.__activations = [[0 for i in range(self.nodes[x])] for x in range(1, len(self.nodes))]
        self.__errors = [[0 for i in range(nodes[x])] for x in range(self.layers)]
        """

        # Will be updated by the forward method
        self.__activations = []

        # Will be updated by the backward method
        self.__errors = []


        print(f"Printing initializedw weights:\n{self.weights}")

        """
        # Adding ones to each activated value since that will become the bias term
        for i in range(self.layers-1):
            self.__activations[i].insert(-1, 1)
        """

        
    def __sigmoid(self, x):
        """ Implementation of sigmoid function"""
        return 1/(1+np.exp(-x))


    def __forward(self, datum):
        """ 
        Forward propagation of finding h_j where h_j is the activated value
        of the j_th node
        """

        data = [datum]
        
        for i in range(len(self.weights)):
            activations = []

            # Only the last layer does not have a bias node
            # This for loop calculates activations starting at the first hidden layer
            if i != len(self.weights) - 1:
                activations = [1]
            for w in range(self.weights[i].shape[1]):
                activations.append(self.__sigmoid(np.dot(data[-1].T, self.weights[i][:, w])))

            data.append(np.asarray(activations))
            self.__activations.append(data[-1])

        return 

    def __backprop(self, target):
        """
        Backpropagation portion of the algorithm
        """
        encoding = [0.9 if x == target else 0.1 for x in range(self.nodes[-1])]
        




        pass


    def train(self):
        for row in range(self.train_data.shape[0]):
            target = self.train_labels[row]
            data = self.train_data[row]

            self.__forward(data)
            print(f"Printing activations:\n{self.__activations}")

            """
            # Work through each layer activating the nodes
            for lr in range(self.layers): 
                for i in range(self.nodes[lr+1]):
                    self.__activations[lr][i] =         \
                             self.__forward(
                                            data, 
                                            self.weights[lr][:, i]
                                           )
                data = np.asarray(self.__activations[lr])
            # At this point all output nodes have some value
            # Now we update weights based on error starting with the last layer before
            # the output layer

            print(f"Target encoding {encoding}")
            print(f"Output nodes    {self.__activations[-1]}")

            # The 2 for loops below will be all incorporated into the self.__weight_update 
            # function but for now doing it here
            for i in range(self.nodes[-1]):
                self.__errors[-1][i] = self.__activations[-1][i] * (1-self.__activations[-1][i]) * \
                                (encoding[i] - self.__activations[-1][i])

            print(f"errors = {self.__errors[1]}")
            print(f"activations = {self.__activations[1]}")
            print(f"weights for output layer = {self.weights[1]}")
            for k in range(self.nodes[1]):
                print(f"K ====== {k}")
                for j in range(self.weights[1].shape[1]):
                    self.weights[1][k, j] = self.weights[1][k, j] + \
                                                 (self.eta * self.__errors[1][j] * \
                                                  self.__activations[1][k]
                                                  )
            print(f"weights for output layer = {self.weights[1]}")
                    

            # Calculating errors for hidden node layer
            for j in range(self.nodes[0]): 
                # Compute sum of the k outputs: w_kj * delta_k
                for k in range(self.outputs):
                    # 1 because we want weights from hidden layer to output 
                    # and in 2 layer node, that is in index=1 in the weights array
                    self.weights[1][j, k]

            print(f"Error array {self.__errors}")
            """

        pass




