import numpy as np


class Perceptron:
    """ Assumed that X has already been scaled prior to being set as parameter """
    def __init__(self, X: np.ndarray, y: np.array, learning_rate=0.01, epochs=100, outputs=1):
        self.X = X
        self.X = self.X 

        self.labels = y

        # One hot encoding the label data
        self.encoded_y = np.zeros((y.size, y.max() + 1)) 
        self.encoded_y[np.arange(y.size), y] = 1

        # Adding bias column to each training data
        ones_column = np.ones((X.shape[0], 1))

        self.X = np.hstack((ones_column, self.X))

        # Hyperparamters
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.outputs = outputs

        # Initializing our weights to be all zeroes
        # Shape of weights matrix is number of outputs times number of paramters in X
        self.weights = np.random.uniform(-0.05, 0.05, (self.outputs, self.X.shape[1]))
        print(f"Dimension of self.weights is {self.weights.shape}")


    def fit(self):
        for i in range(self.epochs):
            # Shuffling data before updating weights
            indices = np.arange(self.X.shape[0])
            np.random.shuffle(indices)
            self.X = self.X[indices]
            self.labels = self.labels[indices]


            # for each row
            for i in range(self.X.shape[0]):

                # Getting predicted class label 
                output_vector = self.weights @ self.X[i].T 
                prediction = np.argmax(output_vector)

                t = 1
                y = 1

                if prediction != self.labels[i]:
                    t = 0
                if output_vector[prediction] <= 0:
                    y = 0

                for row in range(self.weights.shape[0]):
                    self.weights[row] = self.weights[row] + self.learning_rate * (t-y) * self.X[i]

        return self.weights

    def ols(self) -> np.ndarray:  # returns best fit line
        x = np.zeros(self.X.shape[0])
        A_t = np.linalg.matrix_transpose(self.X)
        inv_sym = np.linalg.inv(A_t @ self.X)         # matmul
        half_proj = inv_sym @ A_t
        x = half_proj @ b

        return x




