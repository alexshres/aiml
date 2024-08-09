import numpy as np


class KMeans:
    def __init__(self, training_data, centers=2, r=5):
        self.X = training_data
        self.k = centers
        self.r = r

        mins = self.X.min(axis=0).tolist()
        max = self.X.max(axis=0).tolist()

        self.centers = [] 

        # Randomly initializing centers based on min/max of the data
        for c in range(self.k):
            low = mins[c]
            high = max[c]
            self.centers.append(np.random.uniform(low, high, self.X.shape[1]))

        print(self.centers)



    def train(self):
        pass


    def predict(self, test):
        pass

        

