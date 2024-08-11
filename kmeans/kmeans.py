import numpy as np
import matplotlib.pyplot as plt


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
            low = mins
            high = max
            self.__centers.append(np.random.uniform(low, high, self.X.shape[1]))

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
        min_dist = np.linalg.norm(data - self.__centers[0])

        for i in range(len(self.__centers)):
            curr_dist = np.linalg.norm(data - self.__centers[i])
            # Update min_arg and min_dist only if curr_dist is smaller
            if curr_dist < min_dist:
                min_dist = curr_dist
                min_arg = i

        return min_arg

    def __update_centers(self):
        """
        Updates the self.__centers based on how each data point is clustered
        """

        # Filtering for points assigned to center c
        for c in range(len(self.__centers)):
            points_in_cluster = [self.X[i] for i in range(len(self.__cluster))  \
                    if self.__cluster[i] == c]

            # Updating center to new points based on mean of points in cluster
            if len(points_in_cluster) > 0:
                self.__centers[c] = np.mean(points_in_cluster, axis=0)


    def train(self):
        for e in range(self.__epochs):
            self.__center_size = [0] * len(self.__centers)
            for row in range(self.X.shape[0]):
                # Assign cluster for each data point
                cluster_idx = self.__closest_cluster(self.X[row])
                self.__cluster[row] = cluster_idx
                # Updating number of elements assigned to cluster
                self.__center_size[cluster_idx] += 1
            self.__update_centers()

            if e%2 == 0:
                self.plot_clusters(e)

        # Need to reassign clusters since we have the final center update
        for row in range(self.X.shape[0]):
            # Assign cluster for each data point
            cluster_idx = self.__closest_cluster(self.X[row])
            self.__cluster[row] = cluster_idx
            # Updating number of elements assigned to cluster
            self.__center_size[cluster_idx] += 1

        self.plot_clusters(self.__epochs)

        # Calling sum squared errors
        self.sse()

    def sse(self):
        sse = 0
        for row in range(self.X.shape[0]):
            centroid = self.__centers[self.__cluster[row]]
            distance = np.linalg.norm(self.X[row] - centroid)
            sse += distance ** 2  # Sum of squared distances

        print(f"SSE: {sse}")

    def plot_clusters(self, epoch):
        plt.figure()
        colors = ['r', 'g', 'b', 'y', 'c', 'm']

        # Plotting points based on the cluster they are assigned to
        for i in range(self.k):
            points_in_cluster = np.array([self.X[j] for j in range(len(self.X)) \
                    if self.__cluster[j] == i])

            if points_in_cluster.size > 0:
                plt.scatter(points_in_cluster[:, 0], points_in_cluster[:, 1],   \
                        c=colors[i], label=f"Cluster {i+1}")


        # Plotting the cluster centers
        for i, center in enumerate(self.__centers):
            plt.scatter(center[0], center[1], c='black', marker='x', s=100, \
                    label=f"Center {i+1}")


        plt.title(f"Clustering at Iteration {epoch+1}")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.show()

