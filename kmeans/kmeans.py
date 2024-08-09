import numpy as np


class KMeans:
    def __init__(self, training_data, centers=2, r=5):
        self.X = training_data
        self.k = centers
        self.r = r

        mins = self.X.min(axis=0).tolist()
        max = self.X.max(axis=0).tolist()

        # Initializing centers
        self.__centers = [] 

        # Randomly initializing centers based on min/max of the data
        for c in range(self.k):
            low = mins[c]
            high = max[c]
            self.centers.append(np.random.uniform(low, high, self.X.shape[1]))

        # Defines which cluster each data is assigned to
        # The value is mapped to the index of the self.__centers 
        self.__cluster = [-1] * self.X.shape[0]


    def __closest_cluster(self, data):
        """
        Updates self.__cluster for each data point to the cluster that has the 
        smalles Euclidean distance
        """
        pass

    def __update_centers(self):
        """
        Updates the self.__centers based on how each data point is clustered
        """
        pass


    def train(self):
        # TODO: how to show convergence
        for row in range(self.X.shape[0]):
            self.__closest_cluster(self.X[row])
            self.__update_centers()
        pass


    def predict(self, test):
        pass

        

