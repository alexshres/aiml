import numpy as np
import pandas as pd


class NaiveBayes:
    
    def __init__(self, training_data, training_labels):
        self.X = training_data
        self.y = training_labels

    def __feature_stats(self):
        pass

    def __find_class_prob(self):
        pass

    def __get_priors(self):
        pass

    def train(self):
        pass

    def predict(self, test_data):
        pass
