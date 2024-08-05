import numpy as np
import pandas as pd


class NaiveBayesBinary:
    """
    NaiveBayesBinary class only works for binary classification currently
    """
    
    def __init__(self, training_data, training_labels, num_classes):
        self.X = training_data
        self.y = training_labels
        self.classes = num_classes

        # Grabbing mean and standard deviation
        self.combined_df = pd.concat([self.X, self.y], axis=1)
        self.mean_cls = self.combined_df.groupby('Class').mean()

        # Updating std so no underflow errors
        self.std_cls = self.combined_df.groupby('Class').std()
        self.std_cls = self.std_cls.map(lambda x: 0.0001 if x < 0.0001 else x)

        # Calculating empirical class probabilities
        p_one = self.y.mean()
        p_two = 1 - p_one
        
        self.probabilities = [p_one, p_two]

    def __normal_dist(self, x, mean, std):
        return (1.0 / (std * np.sqrt(2 * np.pi))) *  \
                np.exp(-0.5 * ((x - mean) / std) ** 2)

    def predict(self, test_data):
        predictions = []

        for index, row in test_data.iterrows():
            # Creating likelihoods for each class for each feature
            likelihoods = [0, 0]

            for column in test_data.columns:
                likelihoods[0] += np.log(self.__normal_dist(row[column], 
                                                            self.mean_cls.loc[0, column], 
                                                            self.std_cls.loc[0, column]))

                likelihoods[1] += np.log(self.__normal_dist(row[column], 
                                                            self.mean_cls.loc[1, column], 
                                                            self.std_cls.loc[1, column]))

            # Adding prior probabilities for each class
            for i in range(self.classes):
                likelihoods[i] += np.log(self.probabilities[i])

            predictions.append(np.argmax(likelihoods))

        return predictions


