import numpy as np


class Perceptron:
    """ Assumed that X has already been scaled prior to being set as parameter """
    def __init__(self, inputs, outputs=1, learning_rate=0.01, epochs=100):
        """
        self.X = X
        self.labels = y

        # One hot encoding the label data
        self.encoded_y = np.zeros((y.size, y.max() + 1)) 
        self.encoded_y[np.arange(y.size), y] = 1

        # Adding bias column to each training data
        ones_column = np.ones((X.shape[0], 1))

        self.X = np.hstack((ones_column, self.X))
        """

        self.inputs = inputs
        self.outputs = outputs

        # Hyperparamters
        self.learning_rate = learning_rate
        self.epochs = epochs

        # Initializing our weights to be all zeroes
        # Shape of weights matrix is number of outputs times number of paramters in X
        self.weights = np.random.uniform(-0.05, 0.05, (self.outputs, self.inputs)) 

        self.confusion_matrix = np.zeros((outputs, outputs))




    def train(self, X, y):
        ones_column = np.ones((X.shape[0], 1))
        training_data = np.hstack((ones_column, X))

        labels = y
        encoded_y = np.zeros((y.size, y.max() + 1)) 
        encoded_y[np.arange(y.size), y] = 1

        for i in range(self.epochs):
            # Shuffling data before updating weights
            indices = np.arange(training_data.shape[0])
            np.random.shuffle(indices)
            training_data = training_data[indices]
            labels = labels[indices]


            # for each row
            for i in range(training_data.shape[0]):

                # Getting predicted class label 
                output_vector = self.weights @ training_data[i].T 
                prediction = np.argmax(output_vector)


                for j in range(self.weights.shape[0]):
                    # Updating how far off we were
                    t = encoded_y[i, j]
                    y = 1 if output_vector[j] > 0 else 0
                    self.weights[j] = self.weights[j] + self.learning_rate * \
                            (t-y) * training_data[i]

        return self.weights

    def predict(self, X, y):
        test_data = X
        ones_column = np.ones((X.shape[0], 1))
        test_data = np.hstack((ones_column, test_data))

        test_labels = y
        total = test_data.shape[0]
        correct = 0
        prediction_count = [0 for i in range(10)]
        actual_count = [0 for i in range(10)]

        for i in test_labels:
            actual_count[i] += 1

        for i in range(test_data.shape[0]):
            output = self.weights @ test_data[i].T
            prediction = np.argmax(output)
            prediction_count[prediction] += 1


            if prediction == test_labels[i]:
                correct += 1

        for pred in range(10):
            for act in range(10):
                self.confusion_matrix[pred, act] = (1.0 * prediction_count[pred])/ actual_count[act]



        print(f"Model predicted {correct} out of {total} for a {(1.0 * correct)/total} accuracy")


        print(self.confusion_matrix)



    def ols(self) -> np.ndarray:  # returns best fit line
        x = np.zeros(self.X.shape[0])
        A_t = np.linalg.matrix_transpose(self.X)
        inv_sym = np.linalg.inv(A_t @ self.X)         # matmul
        half_proj = inv_sym @ A_t
        x = half_proj @ b

        return x

