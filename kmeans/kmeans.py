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
        self.__epochs = 10

        # Randomly initializing centers based on min/max of the data
        for c in range(self.k):
            low = mins[c]
            high = max[c]
            self.centers.append(np.random.uniform(low, high, self.X.shape[1]))

        # Defines which cluster each data is assigned to
        # The value is mapped to the index of the self.__centers 
        self.__cluster = [-1] * self.X.shape[0]

        # Number of points mapped to each center
        self.__center_size = [0] * len(self.__centers)



    def __closest_cluster(self, data):
        """
        Returns index of cluster that has the 
        smallest Euclidean distance to the data passed in
        """

        min_arg = 0
        min_dist = np.linalg.norm(data, self.__centers[0])

        for i in range(len(self.__centers)):
            curr_dist = np.linalg.norm(data, self.__centers[i])
            # Update min_arg and min_dist only if curr_dist is smaller
            if curr_dist < min_dist:
                min_dist = curr_dist
                min_arg = i

        return min_arg


    def __update_centers(self):
        """
        Updates the self.__centers based on how each data point is clustered
        """
        
        for c in range(len(self.__centers)):
            temp = np.zeros(self.X.shape[1])

            for i in range(len(self.__clusters)):
                if self.__clusters[i] == c:
                    # Adding up all vectors that are assigned to the vector
                    temp += self.X[i]
             
            # Updating each cluster center
            self.__centers[c] = temp / self.__center_size[c]



    def train(self):
        for e in range(self.__epochs):
            for row in range(self.X.shape[0]):
                # Assign cluster for each data point
                cluster_idx = self.__closest_cluster(self.X[row])
                self.__cluster[row] = cluster_idx
                # Updating number of elements assigned to cluster
                self.__center_size[cluster_idx] += 1
            self.__update_centers()

    def rss(self):




        

